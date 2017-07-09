#!/usr/bin/env roseus

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(ros::load-ros-manifest "jsk_arc2017_baxter")

(unless (find-package "JSK_ARC2017_BAXTER")
  (make-package "JSK_ARC2017_BAXTER"))

(defclass jsk_arc2017_baxter::stow-interface
  :super jsk_arc2017_baxter::arc-interface
  :slots (graspingp
          moveit-p
          picking-fail-count
          label-names
          recognize-fail-count
          target-bin
          target-index
          target-obj
          tote-contents
          grasp-style))

(defmethod jsk_arc2017_baxter::stow-interface
  (:init (&key (moveit nil))
    (send-super :init)
    (setq picking-fail-count 0)
    (setq recognize-fail-count 0)
    (setq moveit-p moveit)
    (setq grasp-style nil))
  (:recognize-bboxes (arm)
    (ros::ros-info "[main] recognizing bin boxes")
    (send self :recognize-bin-boxes :stamp (ros::time-now))
    (ros::ros-info "[main] recognizing tote")
    (send self :recognize-tote-boxes :stamp (ros::time-now))
    (send self :set-movable-region-for-tote arm :offset (list 100 80 0)))
  (:wait-for-user-input (arm)
    (let (can-start)
      (ros::ros-info "[:wait-for-user-input] wait for user input to start: ~a" arm)
      (ros::wait-for-service "/rviz/yes_no_button")
      (setq can-start
            (send (ros::service-call "/rviz/yes_no_button"
                                     (instance jsk_gui_msgs::YesNoRequest))
                  :yes))
      (ros::ros-info "[:wait-for-user-input] received user input: ~a" arm)
      can-start))
  (:recognize-object (arm &key (trial-times 10))
    (let (is-recognized recognition-count)
      (setq label-names
            (ros::get-param (format nil "/~a_hand_camera/label_names"
                                    (arm2str arm))))
      (setq tote-contents (ros::get-param "/tote_contents"))
      (send self :set-target-location arm :tote)
      (when moveit-p (send self :add-shelf-scene))
      (when moveit-p (send self :add-tote-scene arm))
      (send *ri* :stop-grasp arm)
      (send self :move-fingers-pinch->init arm)
      (send *baxter* :head_pan :joint-angle (if (eq arm :larm) -70 70))
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 :head-controller 0)
      (send *ri* :wait-interpolation)
      (ros::ros-info "[main] Recognizing objects in tote")
      ;; TODO: :move-arm-body->tote-overlook-pose
      (send self :move-arm-body->tote-overlook-pose arm)
      (send *ri* :wait-interpolation)
      (setq recognition-count 1)
      (while (null (or (> recognition-count trial-times) is-recognized))
        (setq target-index (send self :get-largest-object-index arm :stamp (ros::time-now)))
        (setq is-recognized
          (when target-index
            (if (not (string= target-obj (elt label-names target-index)))
              ;; reset picking-fail-count for new target
              (setq picking-fail-count 0))
            (setq target-obj (elt label-names target-index))
            (ros::set-dynparam
              (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
              (cons "label_value" target-index))
            (ros::set-param
              (format nil "~a_hand/target_object" (arm2str arm)) target-obj)
            (ros::ros-info-blue "[~a] [main] :recognize-object, target-obj: ~a" (ros::get-name) target-obj)
            (send self :recognize-target-object arm :stamp (ros::time-now))))
        (setq recognition-count (incf recognition-count)))
      is-recognized))
  (:return-from-recognize-object (arm)
    (ros::ros-info "[main] arm: ~a, failed to recognize object in tote" arm)
    (send *ri* :angle-vector-sequence-raw
          (list (send *baxter* :fold-to-keep-object arm)
                (send *baxter* :fold-pose-back arm))
          :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    (when moveit-p (send self :delete-shelf-scene))
    (when moveit-p (send self :delete-tote-scene arm)))
  (:check-recognize-fail-count (arm)
    (incf recognize-fail-count)
    (ros::ros-info "[main] arm: ~a, recognize fail count: ~a" arm recognize-fail-count)
    (> recognize-fail-count 1))
  (:set-grasp-style (arm)
    (if (eq grasp-style :suction)
      (setq grasp-style :pinch) (setq grasp-style :suction)))
  (:pick-object (arm)
    (setq recognize-fail-count 0)
    (when moveit-p (send self :delete-tote-scene arm))
    (send self :reset-scale arm)
    (send self :pick-object-in-tote arm
          :n-trial (if (eq grasp-style :suction) 2 1)
          :n-trial-same-pos 1
          :do-stop-grasp nil
          :grasp-style grasp-style)
    (when moveit-p (send self :add-tote-scene arm))
    (if (eq grasp-style :suction)
      (send *ri* :angle-vector
            (send self :ik->tote-center arm
                  :offset #f(0 0 300) :rotation-axis :z :use-gripper t)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :angle-vector
            (send self :ik->tote-center arm
                  :offset #f(0 0 300) :rotation-axis :z :move-palm-end t)
            3000 (send *ri* :get-arm-controller arm :gripper nil) 0))
    (send *ri* :wait-interpolation)
    (when moveit-p (send self :delete-tote-scene arm))
    (setq graspingp (send *ri* :graspingp arm))
    (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
    graspingp)
  (:return-from-pick-object (arm)
    (send *ri* :stop-grasp arm)
    (ros::ros-info "[main] arm: ~a return from pick-object to fold-pose-back" arm)
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
    ;; TODO: (setq target-obj (send *ti* :recognize-object-in-hand arm))
    ;; TODO: (return-from :verify-objec target-obj)
    (let (scale-candidates)
      (setq scale-candidates
            (send self :get-scale-candidates arm :picked :stamp (ros::time-now)))
      (when (eq scale-candidates :timeout)
        (ros::ros-error "subscribed weight candidates, but timeout")
        (return-from :verify-object nil))
      (ros::ros-info-green "[main] arm ~a: scale candidates ~a" arm scale-candidates)
      (when (eq (length scale-candidates) 1)
        (setq target-obj (car scale-candidates)))
      (if (find target-obj scale-candidates :test #'string=) t nil)))
  (:check-picking-fail-count (arm)
    (incf picking-fail-count)
    (ros::ros-info "[main] arm: ~a, picking fail count: ~a" arm picking-fail-count)
    (> picking-fail-count 1))
  (:set-target-bin (arm)
    ;; TODO: when we select target-bin, we should consider target-obj size.
    ;; TODO: (setq target-bin (send *ti* :get-target-bin arm target-obj))
    (setq target-bin
          (if (eq arm :larm) :a (elt (list :b :c) (random 2))))
    (if (eq target-bin :tote)
      (progn
        (ros::ros-info "[main] unable to recognize, return to tote")
        nil)
      (progn
        (ros::set-param (format nil "~a_hand/target_bin" (arm2str arm)) (symbol2str target-bin))
        (ros::ros-info "[main] target-bin: ~a" target-bin)
        t)))
  (:return-object (arm)
    (if (eq grasp-style :suction)
      (send *ri* :angle-vector-raw
            (send self :ik->tote-center arm :offset #f(0 0 0)
                  :rotation-axis :z :use-gripper t)
            :fast (send *ri* :get-arm-controller arm) 0 :scale 3.0)
      (send *ri* :angle-vector-raw
            (send self :ik->tote-center arm :offset #f(0 0 0)
                  :rotation-axis :z :move-palm-end t)
            :fast (send *ri* :get-arm-controller arm :gripper nil) 0 :scale 3.0))
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp arm)
    (if (eq grasp-style :pinch)
      (send self :move-fingers-pinch->init arm))
    (send self :spin-off-by-wrist arm :times 5)
    (send *ri* :wait-interpolation)
    (ros::ros-info "[main] ~a, return object in tote" arm)
    (when moveit-p (send self :add-tote-scene arm))
    (send self :fold-pose-back arm)
    (send *ri* :wait-interpolation))
  (:place-object (arm)
    (let (dropped)
      (ros::ros-info "[main] ~a, place object in bin ~a" arm target-bin)
      (when moveit-p (send self :add-tote-scene arm))
      (send *baxter* :head_pan :joint-angle (if (eq arm :larm) 70 -70))
      (if (eq grasp-style :suction)
        (send *ri* :angle-vector
              (send self :ik->bin-center arm target-bin
                    :offset #f(0 0 300) :rotation-axis :z :use-gripper t)
              4000 (send *ri* :get-arm-controller arm) 0)
        (send *ri* :angle-vector
              (send self :ik->bin-center arm target-bin
                    :offset #f(0 0 300) :rotation-axis nil :move-palm-end t)
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
          (send self :update-json target-obj :src :tote :dst (cons :bin target-bin))
          (ros::ros-info-green "[main] arm ~a: place object ~a in bin ~a" arm target-obj target-bin)
          (when moveit-p (send self :delete-bin-scene target-bin))
          (send *ri* :angle-vector-raw
                (send *baxter* arm :move-end-pos #f(0 0 -300) :world
                      :rotation-axis (eq grasp-style :suction))
                2000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm) ;; release object
          (if (eq grasp-style :pinch)
            (send *ri* :stop-grasp arm :pinch)) ;; release object
          (send self :spin-off-by-wrist arm :times 10)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector-raw
                (send *baxter* arm :move-end-pos #f(0 0 300) :world
                      :rotation-axis (eq grasp-style :suction))
                1000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (if (eq grasp-style :pinch)
            (send self :move-fingers-pinch->init arm))
          (setq grasp-style nil)
          (when moveit-p (send self :add-bin-scene target-bin))))))
  (:return-from-place-object (arm)
    (send self :fold-pose-back arm)
    (send *ri* :wait-interpolation)))


(defun jsk_arc2017_baxter::stow-init (&key (ctype :default-controller) (moveit nil))
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
      (setq *ti* (instance jsk_arc2017_baxter::stow-interface :init :moveit moveit)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)))
