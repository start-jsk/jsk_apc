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
  (objects (list *baxter*))
  t)

(defun apc-stow-mainloop (arm)
  (ros::ros-info "[main] recognizing bin boxes")
  (send *ri* :recognize-bin-boxes :stamp (ros::time-now))
  ;; TODO recognize order bin
  ;; (ros::ros-info "[main] recognizing order-bin")
  ;; (send *ri* :recognize-order-bin-box :arm arm :stamp (ros::time-now))
  (let (state target-bin)
    (setq state :wait_for_user_input)
    (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
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
        (:wait_for_opposite_arm
          (ros::ros-info "[main] ~a, ~a" arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm-for-stow arm)
            (unix::sleep 1))
          (setq state :pick_object)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:pick_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; if necessary wait for opposite arm
          (if (send *ri* :need-to-wait-opposite-arm-for-stow arm)
            (progn
              (ros::ros-info "[main] Need to wait for opposite arm: ~a" arm)
              (setq state :wait_for_opposite_arm)
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
                          :n-trial 3
                          :n-trial-same-pos 1
                          ;; :object-index (send *ri* :detect-target-object-in-bin target-bin)
                          :do-stop-grasp nil
                          )
                    )
              (setq graspingp t)  ;; don't trust pressure sensor
              (setq state :verify_object)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))))
        ;; verify whether grasping or not
        ;; yes -> set_target_bin to place_object
        ;; no  -> try to pick from order-bin
        (:verify_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
          (if graspingp
            (setq state :set_target_bin)
            (setq state :pick_object))
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:set_target_bin
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq target-bin (send *ri* :select-stow-target-bin arm))
          (if (send *ri* :check-bin-exist target-bin)
              (progn
                (ros::set-param (format nil "~a_hand/target_bin" (send *ri* :arm-symbol2str arm)) (symbol2str target-bin))
                ;; logging
                (ros::ros-info "[main] target-bin: ~a" target-bin)
                (setq state :place_object)
                (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
              (progn
                (ros::ros-error "[main] could not find bin box: ~a" target-bin)
                (setq state :set_target_bin)
                (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))))
        (:place_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] ~a, place object in bin ~a" arm target-bin)
          (send *ri* :move-arm-body->bin arm target-bin)
          (send *ri* :wait-interpolation)
          (setq hcv (send *ri* :get-val '_hard-coded-variables))
          (setq avs-picked->return-bin
                (mapcar
                  #'(lambda (x)
                      (send *ri* :ik->bin-entrance arm target-bin
                            :offset (float-vector x 0 (gethash :offset-avoid-bin-top hcv))))
                  '(-200 -150 -100 -50 0 150)))
          (send *ri* :angle-vector-sequence avs-picked->return-bin :fast nil 0 :scale 5.0)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm) ;; release object
          (send *ri* :spin-off-by-wrist arm :times 20)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector-sequence
                (reverse avs-picked->return-bin)
                :fast nil 0 :scale 5.0)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector (send *baxter* :avoid-shelf-pose arm target-bin))
          (send *ri* :wait-interpolation)
          (send *ri* :fold-pose-back arm)
          (send *ri* :wait-interpolation)
          (setq state :pick_object)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        ))  ;; while & case
    ))

(warn "~% Commands ~%")
(warn "(apc-stow-init)     : initialize *ri*~%")
(warn "(apc-stow-mainloop) : start the mainloop~%~%")
