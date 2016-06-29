#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")

(defun apc-stow-init (&optional (ctype :default-controller))
  (jsk_2016_01_baxter_apc::baxter-init :ctype ctype)
  (send *ri* :gripper-servo-on)
  (send *ri* :angle-vector (send *baxter* :fold-pose-back))
  (send *ri* :wait-interpolation)
  (send *ri* :calib-pressure-threshold)
  (objects (list *baxter*))
  t)

(defun apc-stow-mainloop (arm)
  (ros::ros-info "[main] recognizing bin boxes")
  (send *ri* :recognize-bin-boxes :stamp (ros::time-now))
  (ros::ros-info "[main] recognizing order-bin")
  (send *ri* :recognize-order-bin-box arm :stamp (ros::time-now))
  (send *ri* :get-movable-region-for-tote arm :offset (list 80 80 0))
  (let (state target-bin fail-count)
    (setq state :wait_for_user_input)
    (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
    (setq fail-count 0)
    (while
      t
      (case state
        (:wait_for_user_input
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; wait user input to start the task
          (send *ri* :wait-for-user-input-to-start arm)
          (setq state :initialize)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:initialize
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq state :pick_object)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:wait_for_opposite_arm_in_pick
          (ros::ros-info "[main] ~a, ~a" arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm-for-stow arm :state state)
            (unix::sleep 1))
          (setq state :pick_object)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:pick_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; if necessary wait for opposite arm
          (if (send *ri* :need-to-wait-opposite-arm-for-stow arm :state state)
            (progn
              (ros::ros-info "[main] Need to wait for opposite arm: ~a" arm)
              (setq state :wait_for_opposite_arm_in_pick)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
            (progn
              ;; run motion
              (ros::ros-info "[main] Recognizing objects in tote")
              (send *ri* :move-arm-body->order-bin-overlook-pose arm)
              (send *ri* :wait-interpolation)
              (send *ri* :recognize-objects-in-order-bin :arm arm :stamp (ros::time-now))
              (send *ri* :move-arm-body->order-bin arm)
              (send *ri* :wait-interpolation)
              (send *ri* :gripper-servo-on arm)
              (setq graspingp
                    (send *ri* :pick-object-in-order-bin arm
                          :n-trial 2
                          :n-trial-same-pos 1
                          :do-stop-grasp nil
                          )
                    )
              (setq state :verify_object)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))))
        ;; verify whether grasping or not
        ;;     yes -> verify what the object is
        ;;         get result -> :place_object
        ;;         timeout -> :return_object
        ;;     no  -> :pick_object
        (:verify_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
          (if graspingp
            (progn
              (setq fail-count 0)
              (setq target-obj
                    (send *ri* :recognize-object-in-hand arm
                               :stamp (ros::time-now)
                               :timeout 10000))
              (if (equal target-obj "no_object") (setq target-obj nil))
              (if target-obj
                  (progn
                    (ros::set-param
                      (format nil "~a_hand/target_object"
                              (send *ri* :arm-symbol2str arm)) target-obj)
                    (if (eq arm :rarm) (send *baxter* :view-hand-pose arm :l)
                        (send *baxter* :view-hand-pose arm :j))
                    (send *baxter* arm :move-end-pos #f(-50 0 180) :world)
                    (send *ri* :angle-vector (send *baxter* :angle-vector))
                    (send *ri* :wait-interpolation)
                    (setq object-length
                          (send *ri* :recognize-object-length-in-hand arm
                                :timeout 40000))
                    (setq state :set_target_bin))
                  (setq state :return_object)))
            (progn
              (setq fail-count (incf fail-count))
              (send *ri* :stop-grasp arm)
              (ros::ros-info "[main] arm: ~a, picking fail count: ~a" arm fail-count)
              (if (> fail-count 1)
                (setq state :wait_opposite_arm_start_picking)
                (setq state :wait_for_opposite_arm_in_pick))))
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:wait_opposite_arm_start_picking
          (ros::ros-info "[main] ~a, ~a" arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm-for-stow arm :state state)
            (unix::sleep 1))
          (setq state :wait_for_opposite_arm_in_pick)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:set_target_bin
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq target-bin
                (send *ri* :select-stow-target-bin arm
                      :object-length object-length
                      :object-length-limit 0.1))
          (cond
            ((eq target-bin :tote)
              (ros::ros-info "[main] blacklisted object, return to tote")
              (setq state :return_object)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
            ((send *ri* :check-bin-exist target-bin)
              (ros::set-param (format nil "~a_hand/target_bin" (send *ri* :arm-symbol2str arm)) (symbol2str target-bin))
              ;; logging
              (ros::ros-info "[main] target-bin: ~a" target-bin)
              (setq state :place_object)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
            (t
              (ros::ros-error "[main] could not find bin box: ~a" target-bin)
              (setq state :set_target_bin)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))))
        (:wait_for_opposite_arm_in_place
          (ros::ros-info "[main] ~a, ~a" arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm-for-stow arm :state state)
            (unix::sleep 1))
          (setq state :set_target_bin)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:return_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :move-arm-body->order-bin arm)
          (send *ri* :gripper-servo-off arm)  ;; power off gripper servo not to overload it
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm)
          (send *ri* :spin-off-by-wrist arm :times 20)
          (send *ri* :gripper-servo-on arm)
          (send *ri* :wait-interpolation)
          (send *baxter* :fold-pose-back arm)
          (send *ri* :send-av)
          (send *ri* :wait-interpolation)
          (setq state :pick_object)
          (ros::set-param
            (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
            (symbol2str state)))
        (:place_object
          (if (send *ri* :need-to-wait-opposite-arm-for-stow arm :state state)
            (progn
              (ros::ros-info "[main] Need to wait for opposite arm: ~a" arm)
              (setq state :wait_for_opposite_arm_in_place)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
            (progn
              (ros::ros-info "[main] ~a, ~a" arm state)
              (ros::ros-info "[main] ~a, place object in bin ~a" arm target-bin)
              ;; (setq offset-avoid-bin-top
              ;;       (gethash :offset-avoid-bin-top (send *ri* :get-val '_hard-coded-variables)))
              (if (and object-length (> object-length 0.1))
                (progn
                  (setq entrance-x-list (list -100 -50 0 150 200))
                  (setq offset-avoid-bin-top 80)
                  (setq gripper-angle 45))
                (progn
                  (setq entrance-x-list (list -100 -50 0 150))
                  (setq offset-avoid-bin-top 30)
                  (setq gripper-angle 90)))
              (if (eq arm :larm)
                (send *baxter* :head_pan :joint-angle -80) ;; larm
                (send *baxter* :head_pan :joint-angle 80)  ;; rarm
                )
              (send *ri* :angle-vector-sequence
                    (list
                      (send *baxter* :avoid-shelf-pose arm target-bin)
                      (send *ri* :ik->bin-entrance arm target-bin
                            :offset (float-vector -100 0 offset-avoid-bin-top)
                            :gripper-angle gripper-angle)
                      )
                    :fast nil 0 :scale 5.0)
              (send *ri* :wait-interpolation)
              (setq avs-picked->place-bin
                    (mapcar
                      #'(lambda (x)
                          (send *ri* :ik->bin-entrance arm target-bin
                                :offset (float-vector x 0 offset-avoid-bin-top)
                                :gripper-angle gripper-angle))
                      entrance-x-list))
              (send *ri* :angle-vector-sequence avs-picked->place-bin :fast nil 0 :scale 5.0)
              (send *ri* :wait-interpolation)
              (send *ri* :stop-grasp arm) ;; release object
              (send *ri* :spin-off-by-wrist arm :times 20)
              (send *ri* :wait-interpolation)
              (setq state :return_from_bin)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))))
        (:wait_for_opposite_arm_in_return
          (ros::ros-info "[main] ~a, ~a" arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm-for-stow arm :state state)
            (unix::sleep 1))
          (setq state :return_from_bin)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:return_from_bin
          (if (send *ri* :need-to-wait-opposite-arm-for-stow arm :state state)
            (progn
              (ros::ros-info "[main] Need to wait for opposite arm: ~a" arm)
              (setq state :wait_for_opposite_arm_in_return)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
            (progn
              (ros::ros-info "[main] ~a, ~a" arm state)
              (if (and object-length (> object-length 0.1))
                (setq offset-avoid-bin-top-exit 60)
                (setq offset-avoid-bin-top-exit offset-avoid-bin-top))
              (send *baxter* :rotate-gripper arm 0 :relative nil)
              (setq avs-place-bin->exit
                    (mapcar
                      #'(lambda (x)
                          (send *ri* :ik->bin-entrance arm target-bin
                                :offset (float-vector x 0 offset-avoid-bin-top-exit)
                                :gripper-angle 0))
                      (reverse entrance-x-list)))
              (send *ri* :angle-vector-sequence
                    avs-place-bin->exit
                    :fast nil 0 :scale 5.0)
              (send *ri* :wait-interpolation)
              (send *ri* :angle-vector (send *baxter* :avoid-shelf-pose arm target-bin))
              (send *ri* :wait-interpolation)
              (send *ri* :fold-pose-back arm)
              (send *ri* :wait-interpolation)
              (setq state :pick_object)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))))
        ))  ;; while & case
    ))

(warn "~% Commands ~%")
(warn "(apc-stow-init)     : initialize *ri*~%")
(warn "(apc-stow-mainloop) : start the mainloop~%~%")
