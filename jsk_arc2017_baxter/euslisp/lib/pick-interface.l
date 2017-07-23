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
          graspingp
          grasp-style
          label-names
          moveit-p
          order
          picking-fail-count
          recognize-fail-count
          start-picking-fail-count
          target-bin
          target-cardboard
          target-obj
          trial-fail-count))

(defmethod jsk_arc2017_baxter::pick-interface
  (:init (&key (moveit nil))
    (send-super :init)
    (setq picking-fail-count 0)
    (setq trial-fail-count 0)
    (setq recognize-fail-count 0)
    (setq start-picking-fail-count 0)
    (setq bins (list :a :b :c))
    (setq moveit-p moveit)
    (setq grasp-style nil))
  (:recognize-bboxes ()
    (ros::ros-info "[main] recognizing shelf bin boxes")
    (send self :recognize-bin-boxes :stamp (ros::time-now))
    (ros::ros-info "[main] recognizing cardboard boxes")
    (send self :recognize-cardboard-boxes :stamp (ros::time-now))
    ;; XXX: movable region will be set in :pick-object
    ;; (send self :set-movable-region-for-bin :offset (list 80 120 0))
    )
  (:wait-for-user-input (arm)
    (let (can-start)
      (when moveit-p
        (send self :add-shelf-scene)
        (send self :add-cardboard-rack-scene))
      (ros::ros-info "[:wait-for-user-input] wait for user input to start: ~a" arm)
      (ros::wait-for-service "/rviz/yes_no_button")
      (setq can-start
            (send (ros::service-call "/rviz/yes_no_button"
                                     (instance jsk_gui_msgs::YesNoRequest))
                  :yes))
      (ros::ros-info "[:wait-for-user-input] received user input: ~a" arm)
      (send self :get-work-orders arm)
      (send self :reset-object-tables :bins)
      (dolist (bin bins)
        (send self :set-objects-param arm bin :object-type :all))
      (setq order nil)
      can-start))
  (:set-target (arm)
    (setq label-names
          (ros::get-param (format nil "/~a_hand_camera/label_names"
                                  (arm2str arm))))
    ;; get target param
    (setq target-bin (send self :get-next-target-bin arm))
    (setq bin-contents (send self :get-bin-contents target-bin))
    (unless (send self :check-bin-exist target-bin)
      (ros::ros-warn "[~a] [main] could not find bin box: ~a" (ros::get-name) target-bin)
      (return-from :set-target nil))
    (ros::set-dynparam
      (format nil "/~a_hand_camera/bbox_array_to_bbox" (arm2str arm))
      (cons "index" (position target-bin bins)))
    (send self :set-target-location arm (cons :bin target-bin))
    (ros::set-param
      (format nil "~a_hand/target_bin" (arm2str arm))
      (symbol2str target-bin))
    t)
  (:recognize-object (arm &key (trial-times 10))
    (let (is-recognized recognition-count)
      (setq trial-fail-count 0)
      (setq grasp-style nil)
      (send self :move-fingers-pinch->init arm)
      (ros::ros-info "[main] Recognizing objects in bin ~a" target-bin)
      (unless (> start-picking-fail-count 0)
        (send self :move-arm-body->bin-overlook-pose arm target-bin)
        (send *ri* :wait-interpolation))
      (setq recognition-count 1)
      (while (null (or (> recognition-count trial-times) is-recognized))
        (setq order (send self :select-work-order-in-bin arm target-bin :stamp (ros::time-now)))
        (setq is-recognized
          (when order
            (setq target-obj (send order :item))
            (setq target-cardboard (str2symbol (send order :box)))
            (ros::set-param
              (format nil "~a_hand/target_object" (arm2str arm)) target-obj)
            (ros::set-dynparam
              (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
              (cons "label_value" (position target-obj label-names :test #'string=)))
            (ros::ros-info-blue "[~a] [main] :recognize-object, target-bin: ~a, target-obj: ~a" (ros::get-name) target-bin target-obj)
            (send self :recognize-target-object arm :stamp (ros::time-now))))
        (incf recognition-count))
      is-recognized))
  (:return-from-recognize-object (arm)
    (send self :add-postponed-object arm target-obj target-bin)
    (ros::ros-info "[main] arm: ~a, failed to recognize object ~a" arm target-obj)
    (send *ri* :angle-vector-sequence-raw
          (list (send *baxter* :fold-to-keep-object arm)
                (send *baxter* :fold-pose-back arm))
          :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
    (send *ri* :wait-interpolation))
  (:check-recognize-fail-count (arm)
    (incf recognize-fail-count)
    (ros::ros-info "[main] arm: ~a, recognize fail count: ~a" arm recognize-fail-count)
    (let (recognize-failp)
      (setq recognize-failp (> recognize-fail-count 1))
      (if recognize-failp (setq recognize-fail-count 0))
      recognize-failp))
  (:check-can-start-picking (arm)
    ;; first trial: trial-time = 1
    (setq grasp-style (send self :get-grasp-style arm target-obj 1))
    (ros::ros-info "arm: ~a, grasp-style: ~a, trial-time: ~a"
                   (arm2str arm) (symbol2str grasp-style) (+ trial-fail-count 1))
    (if grasp-style t nil))
  (:check-start-picking-fail-count (arm)
    (incf start-picking-fail-count)
    (ros::ros-info "[main] arm: ~a, start picking fail count: ~a" arm start-picking-fail-count)
    (let (start-picking-failp)
      (setq start-picking-failp (> start-picking-fail-count 1))
      (if start-picking-failp (setq start-picking-fail-count 0))
      start-picking-failp))
  (:change-grasp-style (arm)
    (incf trial-fail-count)
    (send *ri* :stop-grasp arm)
    ;; try another grasp style
    (setq grasp-style (send self :get-grasp-style arm target-obj (+ trial-fail-count 1)))
    (ros::ros-info "arm: ~a, grasp-style: ~a, trial-time: ~a"
                   (arm2str arm) (symbol2str grasp-style) (+ trial-fail-count 1))
    (unless grasp-style (setq trial-fail-count 0))
    (if grasp-style t nil))
  (:pick-object (arm)
    (setq start-picking-fail-count 0)
    (setq recognize-fail-count 0)
    (when moveit-p (send self :delete-bin-scene target-bin))
    (if (eq grasp-style :suction)
      (send self :set-movable-region-for-bin :offset (list 50 50 0))
      (send self :set-movable-region-for-bin :offset (list 50 50 0)))
    (send self :reset-scale arm)
    (send *baxter* :head_pan :joint-angle (if (eq arm :larm) -70 70))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 :head-controller 0)
    (send *ri* :wait-interpolation)
    (send self :pick-object-in-bin arm target-bin
          :n-trial (if (eq grasp-style :suction) 2 1)
          :n-trial-same-pos 1
          :do-stop-grasp nil
          :grasp-style grasp-style)
    (when moveit-p (send self :add-bin-scene target-bin))
    (if (eq grasp-style :suction)
      (send *ri* :angle-vector-raw
            (send self :ik->bin-center arm target-bin
                  :offset #f(0 0 300) :rotation-axis :z :use-gripper t)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :angle-vector-raw
            (send self :ik->bin-center arm target-bin
                  :offset #f(0 0 300) :rotation-axis :z :move-palm-end t)
            3000 (send *ri* :get-arm-controller arm :gripper nil) 0))
    (send *ri* :wait-interpolation)
    (setq graspingp (send *ri* :graspingp arm grasp-style))
    (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
    graspingp)
  (:return-from-pick-object (arm)
    (send self :add-postponed-object arm target-obj target-bin)
    (send *ri* :angle-vector-sequence-raw
          (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))
                (send *baxter* :fold-pose-back arm))
          :fast (send *ri* :get-arm-controller arm :gripper (eq grasp-style :suction)) 0 :scale 4.0)
    (send *ri* :wait-interpolation)
    (if (eq grasp-style :pinch)
      (send self :move-fingers-pinch->init arm)))
  (:verify-object (arm)
    ;; TODO: in-hand verification
    ;; TODO:   target-obj = in-hand object  -> :set_target_cardboard
    ;; TODO:   target-obj != in-hand object -> :return_object
    (let (scale-candidates)
      (setq scale-candidates
            (send self :get-scale-candidates arm :picked :stamp (ros::time-now)))
      (when (eq scale-candidates :timeout)
        (ros::ros-error "subscribed weight candidates, but timeout")
        (return-from :verify-object nil))
      (ros::ros-info-green "[main] arm ~a: scale candidates ~a" arm scale-candidates)
      (if (find target-obj scale-candidates :test #'string=) t nil)))
  (:check-picking-fail-count (arm)
    (incf picking-fail-count)
    (ros::ros-info "[main] arm: ~a, picking fail count: ~a" arm picking-fail-count)
    (let (picking-failp)
      (setq picking-failp (> picking-fail-count 1))
      (if picking-failp (setq picking-fail-count 0))
      picking-failp))
  (:set-target-cardboard (arm)
    (setq picking-fail-count 0)
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
    (send self :add-postponed-object arm target-obj target-bin)
    (when moveit-p (send self :delete-bin-scene target-bin))
    (if (eq grasp-style :suction)
      (send *ri* :angle-vector-raw
            (send self :ik->bin-center arm target-bin
                  :offset #f(0 0 0) :rotation-axis :z :use-gripper t)
            :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
      (send *ri* :angle-vector-raw
            (send self :ik->bin-center arm target-bin
                  :offset #f(0 0 0) :rotation-axis :z :move-palm-end t)
            :fast (send *ri* :get-arm-controller arm :gripper nil) 0 :scale 5.0))
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp arm)
    (if (eq grasp-style :pinch)
      (send self :move-fingers-pinch->init arm))
    (send self :spin-off-by-wrist arm :times 5)
    (send *ri* :wait-interpolation)
    (ros::ros-info "[main] ~a, return object in shelf" arm)
    (send *ri* :angle-vector-raw
          (send *baxter* arm :move-end-pos #f(0 0 200) :world :rotation-axis :z)
          :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    (when moveit-p (send self :add-bin-scene target-bin))
    (send self :fold-pose-back arm)
    (send *ri* :wait-interpolation))
  (:place-object (arm)
    (let (dropped end-coords-name orientation path-constraints orientation-constraint place-z)
      ;; FIXME: moveit cannot solve path
      ;; moveit path constraints
      ;; (setq end-coords-name
      ;;       (if (eq arm :larm) "left_gripper_pad_with_base" "right_gripper_pad_with_base"))
      ;; (setq orientation (instance geometry_msgs::Quaternion :init :x 0 :y 0 :z 0 :w 1))
      ;; (setq orientation-constraint
      ;;       (instance moveit_msgs::OrientationConstraint :init
      ;;                 :link_name end-coords-name
      ;;                 :orientation orientation))
      ;; (send orientation-constraint :header :frame_id end-coords-name)
      ;; (setq path-constraints (instance moveit_msgs::Constraints :init
      ;;                                  :orientation_constraints (list orientation-constraint)))

      (ros::ros-info "[main] ~a, place object in bin ~a" arm target-cardboard)
      (when moveit-p (send self :add-object-in-hand-scene arm target-obj))

      ;; FIXME: moveit cannot solve path
      ;; (send *ri* :angle-vector
      ;;       (send self :ik->cardboard-center arm target-cardboard
      ;;             :offset #f(0 0 200) :rotation-axis :z :use-gripper t)
      ;;       4000 (send *ri* :get-arm-controller arm) 0
      ;;       :path-constraints path-constraints)
      (let (avs)
        (if (eq target-cardboard :c)
          (progn
            (pushback
              (if (eq grasp-style :suction)
                (send self :ik->bin-center arm :b :offset #f(0 0 400)
                      :rotation-axis :z :use-gripper t)
                (send self :ik->bin-center arm :b :offset #f(0 0 400)
                      :rotation-axis nil :move-palm-end t))
              avs)
            (pushback
              (if (eq grasp-style :suction)
                (send self :ik->cardboard-center arm target-cardboard
                      :offset (float-vector 0 0 100) :rotation-axis :z :use-gripper t)
                (send self :ik->cardboard-center arm target-cardboard
                      :offset (float-vector 0 0 100) :rotation-axis nil :move-palm-end t))
              avs)
            (setq place-z 50))
          (progn
            (pushback
              (if (eq grasp-style :suction)
                (send self :ik->bin-center arm :b :offset #f(0 0 800)
                      :rotation-axis :z :use-gripper t)
                (send self :ik->bin-center arm :b :offset #f(0 0 800)
                      :rotation-axis nil :move-palm-end t))
              avs)
            (pushback
              (if (eq grasp-style :suction)
                (send self :ik->cardboard-center arm target-cardboard
                      :offset (float-vector 50 0 200) :rotation-axis :z :use-gripper t)
                (send self :ik->cardboard-center arm target-cardboard
                      :offset (float-vector 50 0 200) :rotation-axis nil :move-palm-end t))
              avs)
            (setq place-z 100)))
        (send *ri* :angle-vector-sequence-raw avs 4000
              (send *ri* :get-arm-controller arm :gripper nil :head t) 0))
      (send *ri* :wait-interpolation)
      (setq dropped (not (send *ri* :graspingp arm grasp-style)))
      (when moveit-p (send self :delete-object-in-hand-scene arm))
      (if dropped
        (progn
          (ros::ros-error "[main] arm ~a: dropped object" arm)
          (send *ri* :stop-grasp arm)
          (if (eq grasp-style :pinch)
            (send self :move-fingers-pinch->init arm))
          (send self :add-postponed-object arm target-obj target-bin))
        (progn
          (ros::ros-info-green "[main] arm ~a: place object ~a in cardboard ~a" arm target-obj target-cardboard)
          (send self :update-json target-obj
                :src (cons :bin target-bin) :dst (cons :cardboard target-cardboard))
          (send self :add-finished-object arm target-obj target-bin)
          (when moveit-p (send self :delete-cardboard-scene target-cardboard))
          (send *ri* :angle-vector-raw
                (if (eq grasp-style :suction)
                  (send *baxter* arm :move-end-pos (float-vector 0 0 (- place-z)) :world)
                  (send *baxter* arm :move-end-pos
                        (float-vector 0 0 (- place-z)) :world :rotation-axis nil))
                2000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm) ;; release object
          (if (eq grasp-style :pinch)
            (send *ri* :stop-grasp arm :pinch)) ;; release object
          (send self :spin-off-by-wrist arm :times 10)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector-raw
                (send *baxter* arm :move-end-pos (float-vector 0 0 place-z) :world
                      :rotation-axis (eq grasp-style :suction))
                1000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (if (eq grasp-style :pinch)
            (send self :move-fingers-pinch->init arm))
          (setq grasp-style nil)
          (when moveit-p (send self :add-cardboard-scene target-cardboard))))))
  (:return-from-place-object (arm)
    (let (avs)
      (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
      (pushback (send *baxter* :fold-pose-back arm) avs)
      (send *ri* :angle-vector-sequence avs 5000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation))))


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
