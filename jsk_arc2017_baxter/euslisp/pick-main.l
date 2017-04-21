#!/usr/bin/env roseus

(ros::roseus "robot_main")

(require "package://jsk_arc2017_baxter/euslisp/lib/baxter-interface.l")

(defun pick-init (&optional (ctype :default-controller) (calib-pressure t))
  (jsk_arc2017_baxter::baxter-init :ctype ctype)
  (send *ri* :gripper-servo-on)
  (send *ri* :angle-vector (send *baxter* :fold-pose-back))
  (send *ri* :wait-interpolation)
  (if calib-pressure
    (send *ri* :calib-pressure-threshold) nil)
  (objects (list *baxter*))
  t)

(defun pick-mainloop (arm)
  (ros::ros-info "[main] recognizing shelf bin boxes")
  (send *ri* :recognize-bin-boxes :stamp (ros::time-now))
  (ros::ros-info "[main] recognizing cardboard boxes")
  (send *ri* :get-movable-region-for-bin :offset (list 60 80 0))

  (setq label-names (ros::get-param (format nil "/~a_hand_camera/label_names" (arm2str arm))))
  (setq bins (list :a :b :c))

  (let (state order target-obj target-cardboard
        fail-count dropped recognition-trial-time is-recognized)
    (setq state :wait_for_user_input)
    (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
    (setq fail-count 0)
    (while t
      (case state
        (:wait_for_user_input
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; wait user input to start the task
          (wait-for-user-input-to-start arm)
          (setq state :initialize)
          (set-arm-state-param arm state))
        (:initialize
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq label-names (ros::get-param (format nil "/~a_hand_camera/label_names" (arm2str arm))))
          (setq state :set_target)
          (set-arm-state-param arm state))
        (:set_target
          ;; next order
          (setq order (send *ri* :get-next-work-order arm order))
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
              (setq bin-contents (get-bin-contents target-bin))
              (ros::set-dynparam
                (format nil "/~a_hand_camera/bbox_array_to_bbox" (arm2str arm))
                (cons "index" (position target-bin bins)))
              (ros::set-dynparam
                (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
                (cons "label_value" (position target-obj label-names :test #'string=)))
              (set-object-segmentation-candidates arm
                (mapcar #'(lambda (x) (position x label-names :test #'string=))
                        (append (list "__background__") bin-contents (list "__shelf__"))))
              (if (send *ri* :check-bin-exist target-bin)
                (progn
                  (ros::set-param
                    (format nil "~a_hand/target_bin" (arm2str arm))
                    (symbol2str target-bin))
                  ;; logging
                  (ros::ros-info-blue "[~a] [main] state: ~a, target-bin: ~a, target-obj: ~a" (ros::get-name) state target-bin target-obj)
                  (setq state :recognize_objects_in_bin))
                (progn
                  (ros::ros-warn "[~a] [main] could not find bin box: ~a" (ros::get-name) target-bin)
                  (setq state :set_target)))))
          (set-arm-state-param arm state))
        (:wait_for_opposite_arm
          (ros::ros-info "[main] ~a, ~a" arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm arm state)
            (unix::sleep 1))
          (setq state :recognize_objects_in_bin)
          (set-arm-state-param arm state))
        (:recognize_objects_in_bin
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; if necessary wait for opposite arm
          (if (send *ri* :need-to-wait-opposite-arm arm state)
            (progn
              (ros::ros-info "[main] Need to wait for opposite arm: ~a" arm)
              (setq state :wait_for_opposite_arm))
            (progn
              (send *baxter* :head_pan :joint-angle (if (eq arm :larm) -80 80))
              (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
              (send *ri* :wait-interpolation)
              (ros::ros-info "[main] Recognizing objects in bin ~a" target-bin)
              (send *ri* :move-arm-body->bin-overlook-pose arm target-bin)
              (send *ri* :wait-interpolation)
              (setq recognition-trial-time 1)
              (setq is-recognized nil)
              (while (null (or (> recognition-trial-time 10) is-recognized))
                (setq is-recognized
                      (send *ri* :recognize-objects-in-bin arm :stamp (ros::time-now)))
                (setq recognition-trial-time (incf recognition-trial-time)))
              (setq state :pick_object)))
          (set-arm-state-param arm state))
        (:pick_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; run motion
          (send *ri* :angle-vector
                (send *ri* :ik->bin-center arm target-bin
                      :offset #f(0 0 100) :coords-pitch 0
                      :rotation-axis :z
                      :use-gripper t)
                3000 (get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (send *ri* :pick-object-in-bin arm target-bin
                :n-trial 2
                :n-trial-same-pos 1
                :do-stop-grasp nil)
          (send *ri* :angle-vector
                (send *ri* :ik->bin-center arm target-bin
                      :offset #f(0 0 200) :coords-pitch 0
                      :rotation-axis :z
                      :use-gripper t)
                3000 (get-arm-controller arm) 0)
          (setq graspingp (send *ri* :graspingp arm))
          (setq state :verify_object)
          (set-arm-state-param arm state))
        (:verify_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
          (if graspingp
            (progn
              (setq fail-count 0)
              (if target-obj
                  (progn
                    (ros::set-param
                      (format nil "~a_hand/target_object"
                              (send *ri* :arm-symbol2str arm)) target-obj)
                    (setq state :set_target_cardboard))
                  (setq state :return_object)))
            (progn
              (setq fail-count (incf fail-count))
              (send *ri* :stop-grasp arm)
              (ros::ros-info "[main] arm: ~a, picking fail count: ~a" arm fail-count)
              (send *ri* :fold-pose-back arm)
              (send *ri* :wait-interpolation)
              (if (> fail-count 1)
                (setq state :wait_for_opposite_arm_start_picking)
                (setq state :wait_for_opposite_arm))))
          (set-arm-state-param arm state))
        (:wait_for_opposite_arm_start_picking
          (ros::ros-info "[main] ~a, ~a" arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm arm state)
            (unix::sleep 1))
          (setq state :wait_for_opposite_arm)
          (set-arm-state-param arm state))
        (:set_target_cardboard
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq dropped nil)
          (if (eq target-cardboard :shelf)
            (progn
              (ros::ros-info "[main] blacklisted object, return to shelf")
              (setq state :return_object))
            (progn
              (ros::set-param (format nil "~a_hand/target_box" (send *ri* :arm-symbol2str arm)) (symbol2str target-cardboard))
              ;; logging
              (ros::ros-info "[main] target-cardboard: ~a" target-cardboard)
              (setq state :place_object)))
          (set-arm-state-param arm state))
        (:return_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :angle-vector
                (send *ri* :ik->bin-center arm target-bin
                      :offset #f(0 0 0) :coords-pitch 0
                      :rotation-axis :z
                      :use-gripper t)
                3000 (get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm)
          (send *ri* :spin-off-by-wrist 5)
          (send *ri* :wait-interpolation)
          (ros::ros-info "[main] ~a, return object in shelf" arm)
          (send *ri* :fold-pose-back arm)
          (send *ri* :wait-interpolation)
          (setq state :set_target)
          (set-arm-state-param arm state))
        (:place_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] ~a, place object in bin ~a" arm target-cardboard)
          (send *baxter* :head_pan :joint-angle (if (eq arm :larm) 80 -80))
          (send *ri* :angle-vector
                (send *ri* :ik->cardboard-entrance arm target-cardboard
                      :offset #f(0 0 200) :rotation-axis :z :use-gripper t)
                3000 (get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector (send *baxter* arm :move-end-pos #f(0 0 -100) :world)
                1000 (get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (setq dropped (not (send *ri* :graspingp arm)))
          (if (not dropped)
            (ros::ros-info-green "[main] arm ~a: place object ~a in cardboard ~a" arm target-obj target-cardboard)
            (ros::ros-warn "[main] arm ~a: dropped object" arm))
          (ros::set-param
            (format nil "~a_hand/target_object"
                    (send *ri* :arm-symbol2str arm)) target-obj)
          (ros::set-param (format nil "~a_hand/target_cardboard" (send *ri* :arm-symbol2str arm)) (symbol2str target-cardboard))
          (send *ri* :stop-grasp arm) ;; release object
          (send *ri* :spin-off-by-wrist arm :times 5)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector (send *baxter* arm :move-end-pos #f(0 0 100) :world)
                1000 (get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (send *ri* :wait-interpolation)
          (setq state :return_from_cardboard)
          (set-arm-state-param arm state))
        (:return_from_cardboard
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :fold-pose-back arm)
          (send *ri* :wait-interpolation)
          (setq state :set_target)
          (set-arm-state-param arm state))))))

(warn "~% Commands ~%")
(warn "(pick-init)     : initialize *ri*~%")
(warn "(pick-mainloop) : start the mainloop~%~%")
