#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/model/pod-lowres.l")

(defun apc-init (&optional (ctype :default-controller))
  (jsk_2015_05_baxter_apc::baxter-init :ctype ctype)
  (objects (list *baxter*))
  t)

(defun apc-mainloop (arm)
  (let (state order target-n-tried target-bin target-obj)
    (setq state :wait_for_user_input)
    (while
      t
      (case
        (:wait_for_user_input
          ;; wait user input to start the task
          (send *ri* :wait-for-user-input-to-start)
          (setq state :initialize)
          (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:initialize
          ;; recognize bin boxes position and dimension
          (send *ri* :recognize-bin-boxes)
          (setq order nil)
          (setq state :set_target)
          (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:set_target
          ;; next order
          (setq order (send *ri* :get-next-work-order arm order))
          ;; get target param
          (setq target-n-tried 0)
          (setq target-bin (send order :bin))
          (setq target-obj (send order :object))
          ;; logging
          (ros::ros-info "[main] state: ~a" state)
          (ros::ros-info "[main] arm: ~a" arm)
          (ros::ros-info "[main] target-bin: ~a" target-bin)
          (ros::ros-info "[main] target-obj: ~a" target-obj)
          (setq state :pick_object)
          (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:wait_for_opposite_arm
          (send *ri* :wait-for-opposite-arm arm)
          (setq state :pick_object)
          (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:pick_object
          ;; if necessary wait for opposite arm
          (when (send *ri* :need-to-wait-opposite-arm arm)
            (setq state :wait_for_opposite_arm)
            (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
            (return-from :pick-object))
          ;; run motion
          (send *ri* :move-arm-body->bin arm target-bin)
          (send *ri* :wait-interpolation)
          (send *ri* :pick-object arm target-bin)
          (incf n-tried)
          (if (= (length (sed *ri* :get-bin-contents target-bin)) 1)
            (setq state :place_object)
            (setq state :move_for_verification))
          (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:move_for_verification
          (send *ri* :move-arm-body->head-view-point arm)
          (send *ri* :wait-interpolation)
          (setq state :verify_object)
          (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:return_object
          (send *ri* :move-arm-body->bin arm target)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector-sequence (send *ri* :ik-avs->bin-entrance arm target :offset #f(300 0 80)))
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp)
          (send *ri* :spin-off-by-wrist arm)
          (send *ri* :wait-interpolation)
          ;; if have tried N times and N is larger than number of bin contents, robot should abort the target
          (if (> (length (send *ri* :get-bin-contents target-bin)) target-n-tried)
            (setq state :pick_object)
            (setq state :set_target) (ros::ros-warn "[main] Abort ~a and go to next target" target-bin))
          (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:place_object
          (send *ri* :move-arm-body->order-bin arm)
          (send *ri* :wait-interpolation)
          (send *ri* :spin-off-by-wrist arm)
          (send *ri* :wait-interpolation)
          (setq state :set_target)
          (ros::set-param (format nil "~a_process/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        ))  ;; while & case
    ))

(warn "~% Commands ~%")
(warn "(apc-init)     : initialize *ri*~%")
(warn "(apc-mainloop) : start the mainloop~%~%")
