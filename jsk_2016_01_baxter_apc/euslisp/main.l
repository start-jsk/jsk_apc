#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")

(defun apc-init (&optional (ctype :default-controller))
  (jsk_2016_01_baxter_apc::baxter-init :ctype ctype)
  (send *ri* :gripper-servo-on)
  (send *ri* :angle-vector (send *baxter* :fold-pose-back))
  (send *ri* :wait-interpolation)
  (send *ri* :calib-pressure-threshold)
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
          (ros::set-param
            (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
            (symbol2str state))
          )
        (:initialize
          (ros::ros-info "[main] ~a, ~a" arm state)
          (setq order nil)
          (setq state :set_target)
          (ros::set-param
            (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
            (symbol2str state))
          )
        (:set_target
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; next order
          (setq order (send *ri* :get-next-work-order arm order))
          (if (null order)
            (progn
              ;; all work orders are done so go to 'wait' status
              (setq state :wait_for_user_input)
              (ros::set-param
                (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
                (symbol2str state))
              )
            (progn
              (ros::ros-warn "next-work-order: ~a" order)
              ;; get target param
              (setq target-n-tried 0)
              (setq target-bin (str2symbol (send order :bin)))
              (setq target-obj (send order :object))
              (if (send *ri* :check-bin-exist target-bin)
                (progn
                  (ros::set-param
                    (format nil "~a_hand/target_bin"
                            (send *ri* :arm-symbol2str arm))
                    (symbol2str target-bin))
                  ;; logging
                  (ros::ros-info "[main] state: ~a" state)
                  (ros::ros-info "[main] arm: ~a" arm)
                  (ros::ros-info "[main] target-bin: ~a" target-bin)
                  (ros::ros-info "[main] target-obj: ~a" target-obj)
                  (setq state :pick_object)
                  (ros::set-param
                    (format nil "~a_hand/state"
                            (send *ri* :arm-symbol2str arm))
                    (symbol2str state)))
                (progn
                  (ros::ros-info "[main] could not find bin box: ~a" target-bin)
                  (setq state :set_target)
                  (ros::set-param
                    (format nil "~a_hand/state"
                            (send *ri* :arm-symbol2str arm))
                    (symbol2str state)))
                )
              )
            )
          )
        (:wait_for_opposite_arm
          (ros::ros-info "[main] ~a, ~a" arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm arm)
            (unix::sleep 1))
          (setq state :pick_object)
          (ros::set-param
            (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
            (symbol2str state))
          )
        (:pick_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          ;; if necessary wait for opposite arm
          (if (send *ri* :need-to-wait-opposite-arm arm)
            (progn
              (ros::ros-info "[main] Need to wait for opposite arm: ~a ~a" arm target-bin)
              (setq state :wait_for_opposite_arm)
              (ros::set-param
                (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
                (symbol2str state)))
            (progn
              ;; run motion
              (ros::ros-info "[main] Recognizing objects in bin" target-bin)
              (send *ri* :move-arm-body->bin-overlook-pose arm target-bin)
              (send *ri* :wait-interpolation)
              (if (send *ri* :recognize-objects-in-bin-loop arm target-bin)
                (progn
                  (setq is-object-found t)
                  ;; (ros::ros-info "detected_object_index: ~A"
                  ;;                (send *ri* :detect-target-object-in-bin target-obj
                  ;;                      target-bin))
                  (setq graspingp
                        (send *ri* :pick-object arm target-bin
                              :n-trial 1
                              :n-trial-same-pos 1
                              ;; :object-index (send *ri* :detect-target-object-in-bin
                              ;;                     target-obj target-bin)
                              :do-stop-grasp t
                              :target-offset #f(50 0 0)
                              )
                        )
                  (incf target-n-tried)
                  )
                (progn
                  (setq graspingp nil)
                  (setq is-object-found nil)
                  )
                )
              (setq state :verify_object)
              (ros::set-param
                (format nil "~a_hand/state"
                        (send *ri* :arm-symbol2str arm))
                (symbol2str state)))
            ))
        (:verify_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
          (if graspingp
            (progn
              (send *baxter* :view-hand-pose arm target-bin)
              (setq avs-picked->view-pose (send *ri* :send-av 5000))
              (send *ri* :wait-interpolation)
              (setq verify-ret (send *ri* :verify-object arm target-obj))
              (ros::ros-info "[main] arm: ~a, verify-ret: ~a" arm verify-ret)
              (cond
                ((not (send self :graspingp arm)) (setq state :return_object))
                ((eq verify-ret :timeout) (setq state :return_object))
                ((null verify-ret) (setq state :return_object))
                (verify-ret (setq state :place_object))
                (t (setq state :return_object))
                )
              ; (if (eq verify-ret :timeout)
              ;   (progn
              ;     ;; there is no object
              ;     (ros::ros-info "[main] arm: ~a, detect no object in hand with vision" arm)
              ;     (send *ri* :stop-grasp arm)
              ;     (send *ri* :angle-vector-sequence
              ;           (list
              ;             (send *baxter* :fold-to-keep-object arm)
              ;             (send *baxter* :fold-pose-back arm)
              ;             )
              ;           :fast nil 0 :scale 5.0)
              ;     ;; if have tried N times and N is larger than
              ;     ;; number of bin contents, robot should abort the target
              ;     (if (> (+ (length (send *ri* :get-bin-contents target-bin)) 1) target-n-tried)
              ;       (setq state :pick_object)
              ;       (setq state :set_target)
              ;       )
              ;     )
              ;   (if verify-ret
              ;     (setq state :place_object)
              ;     (setq state :return_object)
              ;     )
              ;   )
              (ros::set-param
                (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
                (symbol2str state))
              )
            (progn
              (send *ri* :angle-vector-sequence
                    (list
                      (send *baxter* :fold-to-keep-object arm)
                      (send *baxter* :fold-pose-back arm)
                      )
                    :fast nil 0 :scale 5.0)
              ;; if have tried N times and N is larger than
              ;; number of bin contents, robot should abort the target
              (if (and (> (+ (length (send *ri* :get-bin-contents target-bin)) 1) target-n-tried)
                       is-object-found)
                (setq state :pick_object)
                (setq state :set_target)
                )
              )
            )
          )
        (:return_object
          (ros::ros-info "[main] ~a, ~a" arm state)
          (send *ri* :angle-vector-sequence (reverse avs-picked->view-pose) :fast nil 0 :scale 5.0)
          (send *ri* :wait-interpolation)
          (setq hcv (send *ri* :get-val '_hard-coded-variables))
          (setq avs-picked->return-bin
                (cons (send *baxter* :avoid-shelf-pose arm target-bin)
                      (mapcar
                        #'(lambda (x)
                            (send *ri* :ik->bin-entrance arm target-bin
                                  :offset (float-vector x 0 (gethash :offset-avoid-bin-top hcv))))
                        '(-200 -150 -100 -50 0 150))
                      )
                )
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
          ;; if have tried N times and N is larger than
          ;; number of bin contents, robot should abort the target
          (if (> (+ (length (send *ri* :get-bin-contents target-bin)) 1) target-n-tried)
            (setq state :pick_object)
            (setq state :set_target)
            (ros::ros-warn "[main] Abort ~a and go to next target" target-bin))
          (ros::set-param
            (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
            (symbol2str state))
          )
        (:place_object
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
          (setq state :set_target)
          (ros::set-param
            (format nil "~a_hand/state" (send *ri* :arm-symbol2str arm))
            (symbol2str state))
          )
        ))  ;; while & case
    ))

(warn "~% Commands ~%")
(warn "(apc-init)     : initialize *ri*~%")
(warn "(apc-mainloop) : start the mainloop~%~%")
