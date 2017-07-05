#!/usr/bin/env roseus

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(ros::load-ros-manifest "jsk_arc2017_baxter")

(unless (find-package "JSK_ARC2017_BAXTER")
  (make-package "JSK_ARC2017_BAXTER"))

(defclass jsk_arc2017_baxter::pick-interface
  :super jsk_arc2017_baxter::arc-interface
  :slots (bin-contents
          bins
          dropped
          picking-fail-count
          graspingp
          label-names
          moveit-p
          order
          target-bin
          target-cardboard
          target-obj
          grasp-style))

(defmethod jsk_arc2017_baxter::pick-interface
  (:init (&key (moveit nil))
    (send-super :init)
    (setq picking-fail-count 0)
    (setq bins (list :a :b :c))
    (setq moveit-p moveit)
    (setq grasp-style nil))
  (:recognize-bin-boxes ()
    (ros::ros-info "[main] recognizing shelf bin boxes")
    (send-super :recognize-bin-boxes :stamp (ros::time-now))
    (ros::ros-info "[main] recognizing cardboard boxes")
    (send self :set-movable-region-for-bin :offset (list 80 120 0)))
  (:wait-for-user-input (arm)
    (let (can-start)
      (ros::ros-info "[:wait-for-user-input] wait for user input to start: ~a" arm)
      (ros::wait-for-service "/rviz/yes_no_button")
      (setq can-start
            (send (ros::service-call "/rviz/yes_no_button"
                                     (instance jsk_gui_msgs::YesNoRequest))
                  :yes))
      (ros::ros-info "[:wait-for-user-input] received user input: ~a" arm)
      (setq order nil)
      can-start))
  (:set-target (arm)
    ;; reset picking-fail-count for new target
    (setq picking-fail-count 0)
    (setq label-names
          (ros::get-param (format nil "/~a_hand_camera/label_names"
                                  (arm2str arm))))
    (setq order (send self :get-next-work-order arm order))
    (unless order (return-from :set-target nil))
    (ros::ros-warn "next-work-order: ~a" (send order :item))
    ;; get target param
    (setq target-bin (str2symbol (send order :bin)))
    (setq target-obj (send order :item))
    (setq target-cardboard (str2symbol (send order :box)))
    (setq bin-contents (send self :get-bin-contents target-bin))
    (unless (send self :check-bin-exist target-bin)
      (ros::ros-warn "[~a] [main] could not find bin box: ~a" (ros::get-name) target-bin)
      (return-from :set-target nil))
    (ros::set-param
      (format nil "~a_hand/target_object" (arm2str arm)) target-obj)
    (ros::set-dynparam
      (format nil "/~a_hand_camera/bbox_array_to_bbox" (arm2str arm))
      (cons "index" (position target-bin bins)))
    (ros::set-dynparam
      (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
      (cons "label_value" (position target-obj label-names :test #'string=)))
    (send self :set-object-segmentation-candidates arm
      (mapcar #'(lambda (x) (position x label-names :test #'string=))
              (append (list "__background__") bin-contents (list "__shelf__"))))
    (ros::set-param
      (format nil "~a_hand/target_bin" (arm2str arm))
      (symbol2str target-bin))
    ;; logging
    (ros::ros-info-blue "[~a] [main] :set-target, target-bin: ~a, target-obj: ~a" (ros::get-name) target-bin target-obj)
    t)
  (:recognize-object (arm &key (trial-times 10))
    (let (is-recognized recognition-count)
      (when moveit-p (send self :add-shelf-scene))
      (when moveit-p (send self :add-cardboard-scene))
      (send *ri* :stop-grasp arm)
      (if (eq arm :rarm)
        (send self :move-fingers-pinch->init arm))
      (send *baxter* :head_pan :joint-angle (if (eq arm :larm) -80 80))
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 :head-controller 0)
      (send *ri* :wait-interpolation)
      (ros::ros-info "[main] Recognizing objects in bin ~a" target-bin)
      (send self :move-arm-body->bin-overlook-pose arm target-bin)
      (send *ri* :wait-interpolation)
      (setq recognition-count 1)
      (while (null (or (> recognition-count trial-times) is-recognized))
        (setq is-recognized
              (send self :recognize-target-object arm :stamp (ros::time-now)))
        (setq recognition-count (incf recognition-count)))
      is-recognized))
  (:return-from-recognize-object (arm)
    (ros::ros-info "[main] arm: ~a, failed to recognize object ~a" arm target-obj)
    (send *ri* :angle-vector-sequence-raw
          (list (send *baxter* :fold-to-keep-object arm)
                (send *baxter* :fold-pose-back arm))
          :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    (when moveit-p (send self :delete-shelf-scene))
    (when moveit-p (send self :delete-cardboard-scene)))
  (:set-grasp-style (arm)
    (if (and (eq grasp-style :suction) (eq arm :rarm))
      (setq grasp-style :pinch) (setq grasp-style :suction)))
  (:pick-object (arm)
    (when moveit-p (send self :delete-shelf-scene))
    (send self :pick-object-in-bin arm target-bin
          :n-trial 1
          :n-trial-same-pos 1
          :do-stop-grasp nil
          :grasp-style grasp-style)
    (when moveit-p (send self :add-shelf-scene))
    (if (eq grasp-style :suction)
      (send *ri* :angle-vector
            (send self :ik->bin-center arm target-bin
                  :offset #f(0 0 300) :rotation-axis :z :use-gripper t)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :angle-vector
            (send self :ik->bin-center arm target-bin
                  :offset #f(0 0 300) :rotation-axis :z :move-palm-end t)
            3000 (send *ri* :get-arm-controller arm :gripper nil) 0))
    (send *ri* :wait-interpolation)
    (when moveit-p (send self :delete-shelf-scene))
    (setq graspingp (send *ri* :graspingp arm))
    (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
    graspingp)
  (:return-from-pick-object (arm)
    (send *ri* :stop-grasp arm)
    (send *ri* :angle-vector-sequence-raw
          (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))
                (send *baxter* :fold-pose-back arm))
          :fast (send *ri* :get-arm-controller arm :gripper (eq grasp-style :suction)) 0 :scale 4.0)
    (send *ri* :wait-interpolation)
    (if (eq grasp-style :pinch)
      (send self :move-fingers-pinch->init arm)))
  (:verify-object (arm)
    (setq picking-fail-count 0)
    ;; TODO: in-hand verification
    ;; TODO:   target-obj = in-hand object  -> :set_target_cardboard
    ;; TODO:   target-obj != in-hand object -> :return_object
    t)
  (:check-picking-fail-count (arm)
    (setq picking-fail-count (incf picking-fail-count))
    (ros::ros-info "[main] arm: ~a, picking fail count: ~a" arm picking-fail-count)
    (> picking-fail-count 1))
  (:set-target-cardboard (arm)
    (if (eq target-cardboard :shelf)
      (progn
        (ros::ros-info "[main] blacklisted object, return to shelf")
        nil)
      (progn
        (ros::set-param (format nil "~a_hand/target_box" (arm2str arm)) (symbol2str target-cardboard))
        ;; logging
        (ros::ros-info "[main] target-cardboard: ~a" target-cardboard)
        t)))
  (:return-object (arm)
    (if (eq grasp-style :suction)
      (send *ri* :angle-vector-raw
            (send self :ik->bin-center arm target-bin
                  :offset #f(0 0 0) :rotation-axis :z :use-gripper t)
            :fast (send *ri* :get-arm-controller arm) 0 :scale 3.0)
      (send *ri* :angle-vector-raw
            (send self :ik->bin-center arm target-bin
                  :offset #f(0 0 0) :rotation-axis :z :move-palm-end t)
            :fast (send *ri* :get-arm-controller arm :gripper nil) 0 :scale 3.0))
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp arm)
    (if (eq grasp-style :pinch)
      (send self :move-fingers-pinch->init arm))
    (send self :spin-off-by-wrist 5)
    (send *ri* :wait-interpolation)
    (ros::ros-info "[main] ~a, return object in shelf" arm)
    (when moveit-p (send self :add-shelf-scene))
    (send self :fold-pose-back arm)
    (send *ri* :wait-interpolation))
  (:place-object (arm)
    (let (dropped end-coords-name orientation path-constraints orientation-constraint)
      ;; moveit path constraints
      (setq end-coords-name
            (if (eq arm :larm) "left_gripper_vacuum_pad" "right_gripper_pad_with_base"))
      (setq orientation (instance geometry_msgs::Quaternion :init :x 0 :y 0 :z 0 :w 1))
      (setq orientation-constraint
            (instance moveit_msgs::OrientationConstraint :init
                      :link_name end-coords-name
                      :orientation orientation))
      (send orientation-constraint :header :frame_id end-coords-name)
      (setq path-constraints (instance moveit_msgs::Constraints :init
                                       :orientation_constraints (list orientation-constraint)))

      (ros::ros-info "[main] ~a, place object in bin ~a" arm target-cardboard)
      (when moveit-p (send self :add-shelf-scene))
      (send *baxter* :head_pan :joint-angle (if (eq arm :larm) 80 -80))
      (if (eq grasp-style :suction)
        (send *ri* :angle-vector
              (send self :ik->cardboard-center arm target-cardboard
                    :offset #f(0 0 200) :rotation-axis :z :use-gripper t)
              4000 (send *ri* :get-arm-controller arm) 0
              :path-constraints path-constraints)
        ;; TODO Add path constraints to pinching
        (send *ri* :angle-vector
              (send self :ik->cardboard-center arm target-cardboard
                    :offset #f(0 0 350) :rotation-axis nil :move-palm-end t)
              4000 (send *ri* :get-arm-controller arm :gripper nil) 0))
      (send *ri* :wait-interpolation)
      (setq dropped (not (send *ri* :graspingp arm)))
      (if dropped
        (progn
          (ros::ros-error "[main] arm ~a: dropped object" arm)
          (send *ri* :stop-grasp arm)
          (if (eq grasp-style :pinch)
            (send self :move-fingers-pinch->init arm)))
        (progn
          (ros::ros-info-green "[main] arm ~a: place object ~a in cardboard ~a" arm target-obj target-cardboard)
          (send self :update-json target-obj
                :src (cons :bin target-bin) :dst (cons :cardboard target-cardboard))
          (when moveit-p (send self :delete-cardboard-scene))
          (send *ri* :angle-vector-raw
                (if (eq grasp-style :suction)
                  (send *baxter* arm :move-end-pos #f(0 0 -200) :world)
                  (send *baxter* arm :move-end-pos #f(0 0 -350) :world :rotation-axis nil))
                2000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm) ;; release object
          (if (eq grasp-style :pinch)
            (send *ri* :stop-grasp arm :pinch)) ;; release object
          (send self :spin-off-by-wrist arm :times 10)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector-raw
                (send *baxter* arm :move-end-pos #f(0 0 200) :world
                      :rotation-axis (eq grasp-style :suction))
                1000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (if (eq grasp-style :pinch)
            (send self :move-fingers-pinch->init arm))
          (setq grasp-style nil)
          (when moveit-p (send self :add-cardboard-scene))))))
  (:return-from-place-object (arm)
    (send self :fold-pose-back arm)
    (send *ri* :wait-interpolation)))


(defun jsk_arc2017_baxter::pick-init (&key (ctype :default-controller) (moveit nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance jsk_arc2017_baxter::baxter-moveit-environment))
      (setq mvit-rb (instance jsk_arc2017_baxter::baxter-robot :init)))
    (unless (boundp '*ri*)
      (setq *ri* (instance jsk_arc2017_baxter::baxter-interface :init :type ctype
                           :moveit-environment mvit-env
                           :moveit-robot mvit-rb)))
    (unless (boundp '*baxter*)
      (setq *baxter* (instance jsk_arc2017_baxter::baxter-robot :init)))
    (unless (boundp '*co*)
      (setq *co* (when moveit (instance collision-object-publisher :init))))
    (unless (boundp '*ti*)
      (setq *ti* (instance jsk_arc2017_baxter::pick-interface :init :moveit moveit)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)))
