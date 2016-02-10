#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")

(defun apc-init (&optional (ctype :default-controller))
  (jsk_2015_05_baxter_apc::baxter-init :ctype ctype)
;  (send *ri* :angle-vector (send *baxter* :fold-pose-back))
;  (send *ri* :wait-interpolation)
  (objects (list *baxter*))
  t)

(defun apc-mainloop (arm)
  ;; recognize bin boxes position and dimension
  (ros::ros-info "[main] recognizing bin boxes")
  (send *ri* :recognize-bin-boxes :stamp (ros::time-now))
  (let (state order target-n-tried target-bin target-obj)
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
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:initialize
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq order nil)
          (setq state :set_target)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:set_target
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; next order
          (setq order (send *ri* :get-next-work-order arm order))
          (if (null order)
            (progn
              ;; all work orders are done so go to 'wait' status
              (setq state :wait_for_user_input)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
              )
            (progn
              (ros::ros-warn "next-work-order: ~a" order)
              ;; get target param
              (setq target-n-tried 0)
              (setq target-bin (str2symbol (send order :bin)))
              (setq target-obj (send order :object))
              (ros::set-param (format nil "~a_hand/target_bin" (send *ri* :arm-symbol2str arm)) (symbol2str target-bin))
              ;; logging
              (ros::ros-info "[main] state: ~a" state)
              (ros::ros-info "[main] arm: ~a" arm)
              (ros::ros-info "[main] target-bin: ~a" target-bin)
              (ros::ros-info "[main] target-obj: ~a" target-obj)
              (setq state :pick_object)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
              )
            )
          )
        (:wait_for_opposite_arm
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :wait-for-opposite-arm arm)
          (setq state :pick_object)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:pick_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; if necessary wait for opposite arm
          (if (send *ri* :need-to-wait-opposite-arm arm)
            (progn
              (ros::ros-info "[main] Need to wait for opposite arm: ~a ~a" arm target-bin)
              (setq state :wait_for_opposite_arm)
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
            (progn
              ;; run motion
              (ros::ros-info "[main] Recognizing objects in bin" target-bin)
              (send *ri* :recognize-objects-in-bin target-bin :stamp (ros::time-now))
              (send *ri* :move-arm-body->bin arm target-bin)
              (send *ri* :wait-interpolation)
              (send *ri* :pick-object arm target-bin :n-trial 2 :n-trial-same-pos 2)
              (incf target-n-tried)
              (if (= (length (send *ri* :get-bin-contents target-bin)) 1)
                (setq state :place_object)
                (setq state :move_for_verification))
              (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
            ))
        (:move_for_verification
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :move-arm-body->head-view-point arm)
          (send *ri* :wait-interpolation)
          (setq state :verify_object)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:verify_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (if (send *ri* :verify-object arm target-obj)
            (setq state :place_object)
            (setq state :return_object))
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state)))
        (:return_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :move-arm-body->bin arm target-bin)
          (send *ri* :wait-interpolation)
          (send *ri* :ik->bin-entrance arm target-bin :offset #f(300 0 50))
          (send *ri* :send-av)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp) ;; release object
          (send *ri* :spin-off-by-wrist arm)
          (send *ri* :wait-interpolation)
          (send *ri* :ik->bin-entrance arm target-bin) ;; to bin entrance
          (send *ri* :send-av 5000)
          (send *ri* :wait-interpolation)
          (send *baxter* :avoid-shelf-pose arm target-bin)
          (send *ri* :send-av)
          (send *ri* :wait-interpolation)
          (send *ri* :fold-pose-back arm)
          ;; if have tried N times and N is larger than number of bin contents, robot should abort the target
          (if (> (length (send *ri* :get-bin-contents target-bin)) target-n-tried)
            (setq state :pick_object)
            (setq state :set_target) (ros::ros-warn "[main] Abort ~a and go to next target" target-bin))
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        (:place_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :move-arm-body->order-bin arm)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm)
          (send *ri* :spin-off-by-wrist arm :times 20)
          (send *ri* :wait-interpolation)
          (send *baxter* :fold-pose-back arm)
          (send *ri* :send-av)
          (send *ri* :wait-interpolation)
          (setq state :set_target)
          (ros::set-param (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm)) (symbol2str state))
          )
        ))  ;; while & case
    ))

(warn "~% Commands ~%")
(warn "(apc-init)     : initialize *ri*~%")
(warn "(apc-mainloop) : start the mainloop~%~%")
