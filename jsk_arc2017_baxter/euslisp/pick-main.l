#!/usr/bin/env roseus

(ros::roseus "robot_main")

(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(defun pick-init (&key (ctype :default-controller) (calib-pressure t) (moveit nil))
  (jsk_arc2017_baxter::arc-init :ctype ctype :moveit moveit)
  (when moveit (send *arc* :wipe-all-scene))
  (send *ri* :gripper-servo-on)
  (send *ri* :angle-vector (send *baxter* :fold-pose-back))
  (send *ri* :wait-interpolation)
  (when calib-pressure
    (send *ri* :calib-pressure-threshold :rarm)
    (send *ri* :calib-pressure-threshold :larm))
  (objects (list *baxter*))
  t)

(defun pick-mainloop (arm)
  (ros::ros-info "[main] recognizing shelf bin boxes")
  (send *arc* :recognize-bin-boxes :stamp (ros::time-now))
  (ros::ros-info "[main] recognizing cardboard boxes")
  (send *arc* :set-movable-region-for-bin :offset (list 80 120 0))

  (setq label-names (ros::get-param (format nil "/~a_hand_camera/label_names" (arm2str arm))))
  (setq bins (list :a :b :c))

  (let (state order target-obj target-cardboard
        fail-count dropped recognition-trial-time is-recognized)
    (setq state :wait_for_user_input)
    (send *arc* :set-arm-state-param arm state)
    (setq fail-count 0)
    (while t
      (case state
        (:wait_for_user_input
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; wait user input to start the task
          (send *arc* :wait-for-user-input-to-start arm)
          (setq state :initialize)
          (send *arc* :set-arm-state-param arm state))
        (:initialize
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq label-names (ros::get-param (format nil "/~a_hand_camera/label_names" (arm2str arm))))
          (setq state :set_target)
          (send *arc* :set-arm-state-param arm state))
        (:set_target
          ;; next order
          (setq order (send *arc* :get-next-work-order arm order))
          (if (null order)
            (progn
              ;; all work orders are done so go to 'wait' status
              (setq state :wait_for_user_input))
            (progn
              (ros::ros-warn "next-work-order: ~a" (send order :item))
              ;; get target param
              (setq target-bin (str2symbol (send order :bin)))
              (setq target-obj (send order :item))
              (setq target-cardboard (str2symbol (send order :box)))
              (setq bin-contents (send *arc* :get-bin-contents target-bin))
              (ros::set-param
                (format nil "~a_hand/target_object" (arm2str arm)) target-obj)
              (ros::set-dynparam
                (format nil "/~a_hand_camera/bbox_array_to_bbox" (arm2str arm))
                (cons "index" (position target-bin bins)))
              (ros::set-dynparam
                (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
                (cons "label_value" (position target-obj label-names :test #'string=)))
              (send *arc* :set-object-segmentation-candidates arm
                (mapcar #'(lambda (x) (position x label-names :test #'string=))
                        (append (list "__background__") bin-contents (list "__shelf__"))))
              (if (send *arc* :check-bin-exist target-bin)
                (progn
                  (ros::set-param
                    (format nil "~a_hand/target_bin" (arm2str arm))
                    (symbol2str target-bin))
                  ;; logging
                  (ros::ros-info-blue "[~a] [main] state: ~a, target-bin: ~a, target-obj: ~a" (ros::get-name) state target-bin target-obj)
                  (setq state :wait_for_opposite_arm))
                (progn
                  (ros::ros-warn "[~a] [main] could not find bin box: ~a" (ros::get-name) target-bin)
                  (setq state :set_target)))))
          (send *arc* :set-arm-state-param arm state))
        (:wait_for_opposite_arm
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] Need to wait for opposite arm: ~a" arm)
          (while
            (send *arc* :need-to-wait-opposite-arm arm state)
            (unix::sleep 1))
          (setq state :recognize_objects_in_bin)
          (send *arc* :set-arm-state-param arm state))
        (:recognize_objects_in_bin
          (send *arc* :add-shelf-scene)
          (send *arc* :add-cardboard-scene arm)
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *baxter* :head_pan :joint-angle (if (eq arm :larm) -80 80))
          (ros::ros-info "[main] Recognizing objects in bin ~a" target-bin)
          (send *arc* :move-arm-body->bin-overlook-pose arm target-bin)
          (send *ri* :wait-interpolation)
          (setq recognition-trial-time 1)
          (setq is-recognized nil)
          (while (null (or (> recognition-trial-time 10) is-recognized))
            (setq is-recognized
                  (send *arc* :recognize-objects-in-bin arm :stamp (ros::time-now)))
            (setq recognition-trial-time (incf recognition-trial-time)))
          (if is-recognized
            (setq state :pick_object)
            (progn
              (ros::ros-info "[main] arm: ~a, failed to recognize object ~a" arm target-obj)
              (send *ri* :angle-vector-sequence
                    (list (send *baxter* :fold-to-keep-object arm)
                          (send *baxter* :fold-pose-back arm))
                    :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
              (send *ri* :wait-interpolation)
              (send *arc* :delete-shelf-scene)
              (send *arc* :delete-cardboard-scene arm)
              (setq state :wait_for_opposite_arm_start_picking)))
          (send *arc* :set-arm-state-param arm state))
        (:pick_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *arc* :delete-shelf-scene)
          (send *arc* :pick-object-in-bin arm target-bin
                :n-trial 2
                :n-trial-same-pos 1
                :do-stop-grasp nil)
          (send *ri* :angle-vector
                (send *arc* :ik->bin-center arm target-bin
                      :offset #f(0 0 300) :rotation-axis :z :use-gripper t)
                3000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (setq graspingp (send *ri* :graspingp arm))
          (setq state :verify_object)
          (send *arc* :set-arm-state-param arm state))
        (:verify_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
          (if graspingp
            (progn
              (setq fail-count 0)
              ;; TODO: in-hand verification
              ;; TODO:   target-obj = in-hand object  -> :set_target_cardboard
              ;; TODO:   target-obj != in-hand object -> :return_object
              (if target-obj
                  (setq state :set_target_cardboard)
                  (setq state :return_object)))
            (progn
              (setq fail-count (incf fail-count))
              (send *ri* :stop-grasp arm)
              (ros::ros-info "[main] arm: ~a, picking fail count: ~a" arm fail-count)
              (send *ri* :angle-vector-sequence
                    (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))
                          (send *baxter* :fold-pose-back arm))
                    :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
              (send *ri* :wait-interpolation)
              (if (> fail-count 1)
                (setq state :wait_for_opposite_arm_start_picking)
                (setq state :wait_for_opposite_arm))))
          (send *arc* :set-arm-state-param arm state))
        (:wait_for_opposite_arm_start_picking
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] Waiting for opposite arm start picking: ~a" arm)
          (while
            (send *arc* :need-to-wait-opposite-arm arm state)
            (unix::sleep 1))
          (setq state :wait_for_opposite_arm)
          (send *arc* :set-arm-state-param arm state))
        (:set_target_cardboard
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq dropped nil)
          (if (eq target-cardboard :shelf)
            (progn
              (ros::ros-info "[main] blacklisted object, return to shelf")
              (setq state :return_object))
            (progn
              (ros::set-param (format nil "~a_hand/target_box" (arm2str arm)) (symbol2str target-cardboard))
              ;; logging
              (ros::ros-info "[main] target-cardboard: ~a" target-cardboard)
              (setq state :place_object)))
          (send *arc* :set-arm-state-param arm state))
        (:return_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :angle-vector
                (send *arc* :ik->bin-center arm target-bin
                      :offset #f(0 0 0) :coords-pitch 0
                      :rotation-axis :z
                      :use-gripper t)
                :fast (send *ri* :get-arm-controller arm) 0 :scale 3.0)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm)
          (send *ri* :spin-off-by-wrist 5)
          (send *ri* :wait-interpolation)
          (ros::ros-info "[main] ~a, return object in shelf" arm)
          (send *arc* :add-shelf-scene)
          (send *ri* :fold-pose-back arm)
          (send *ri* :wait-interpolation)
          (setq state :set_target)
          (send *arc* :set-arm-state-param arm state))
        (:place_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] ~a, place object in bin ~a" arm target-cardboard)
          (send *arc* :add-shelf-scene)
          (send *baxter* :head_pan :joint-angle (if (eq arm :larm) 80 -80))
          (send *ri* :angle-vector
                (send *arc* :ik->cardboard-entrance arm target-cardboard
                      :offset #f(0 0 300) :rotation-axis :z :use-gripper t)
                4000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (setq dropped (not (send *ri* :graspingp arm)))
          (if dropped
            (progn
              (ros::ros-error "[main] arm ~a: dropped object" arm)
              (send *ri* :stop-grasp arm))
            (progn
              (ros::ros-info-green "[main] arm ~a: place object ~a in cardboard ~a" arm target-obj target-cardboard)
              (send *arc* :delete-cardboard-scene arm)
              (send *ri* :angle-vector-raw (send *baxter* arm :move-end-pos #f(0 0 -300) :world)
                    2000 (send *ri* :get-arm-controller arm) 0)
              (send *ri* :wait-interpolation)
              (send *ri* :stop-grasp arm) ;; release object
              (send *ri* :spin-off-by-wrist arm :times 10)
              (send *ri* :wait-interpolation)
              (send *ri* :angle-vector-raw (send *baxter* arm :move-end-pos #f(0 0 300) :world)
                    1000 (send *ri* :get-arm-controller arm) 0)
              (send *ri* :wait-interpolation)
              (send *arc* :add-cardboard-scene arm)))
          (setq state :return_from_cardboard)
          (send *arc* :set-arm-state-param arm state))
        (:return_from_cardboard
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :fold-pose-back arm)
          (send *ri* :wait-interpolation)
          (setq state :set_target)
          (send *arc* :set-arm-state-param arm state))))))

(warn "~% Commands ~%")
(warn "(pick-init)     : initialize *arc*~%")
(warn "(pick-mainloop) : start the mainloop~%~%")
