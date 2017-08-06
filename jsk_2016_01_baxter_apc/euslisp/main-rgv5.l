#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxterrgv5-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")

(defun apc-init (&optional (ctype :default-controller) (calib-pressure t))
  (jsk_2016_01_baxter_apc::baxterrgv5-init :ctype ctype)
  (send *ri* :gripper-servo-on)
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pre-pose :rarm :cylindrical) 1500)
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pre-pose :rarm :opposed) 1500)
  (send *baxter* :fold-pose-back)
  ;; view kiva pose
  (send *baxter* :larm :shoulder-y :joint-angle 70)
  (send *baxter* :rarm :shoulder-y :joint-angle -70)
  (send *baxter* :larm :move-end-pos #f(0 0 100))
  (send *baxter* :rarm :move-end-pos #f(0 0 100))
  (send *ri* :angle-vector (send *baxter* :angle-vector) 10000)
  (send *ri* :wait-interpolation)
  (if calib-pressure
    (send *ri* :calib-pressure-threshold) nil)
  (objects (list *baxter*))
  t)

(defun apc-mainloop (arm)
  ;; recognize bin boxes position and dimension
  (ros::ros-info-blue "[~a] [main] recognizing bin boxes" (ros::get-name))
  (send *ri* :recognize-bin-boxes :stamp (ros::time-now))

  (setq label-names (ros::get-param (format nil "/~a_hand_camera/label_names" (arm2str arm))))
  (setq bins (list :a :b :c :d :e :f :g :h :i :j :k :l))

  (let (state order past-graspingps target-bin target-obj object-found-p)
    (setq state :wait_for_user_input)
    (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
    (while
      t
      (case state
        (:wait_for_user_input
          (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
          (ros::ros-info-blue "[~a] [main] ~a, ~a" (ros::get-name) arm state)
          ;; wait user input to start the task
          (send *ri* :wait-for-user-input-to-start arm)
          (setq state :initialize)
          )
        (:initialize
          (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
          (ros::ros-info-blue "[~a] [main] ~a, ~a" (ros::get-name) arm state)
          (send *ri* :gripper-servo-on arm)
          (send *ri* :angle-vector (send *baxter* :fold-pose-back) 3000)
          (send *ri* :wait-interpolation)
          (setq order nil)
          (setq state :set_target)
          )
        (:set_target
          (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
          (ros::ros-info-blue "[~a] [main] ~a, ~a" (ros::get-name) arm state)
          ;; next order
          (setq order (send *ri* :get-next-work-order arm order))
          (if (null order)
            (progn
              ;; all work orders are done so go to 'wait' status
              (setq state :wait_for_user_input)
              )
            (progn
              (ros::ros-warn "next-work-order: ~a" order)
              ;; get target param
              (setq past-graspingps nil)
              (setq target-bin (str2symbol (send order :bin)))
              (setq target-obj (send order :object))
              (setq bin-contents (send *ri* :get-bin-contents target-bin))
              (ros::set-dynparam
                (format nil "/~a_hand_camera/bbox_array_to_bbox" (arm2str arm))
                (cons "index" (position target-bin bins)))
              (ros::set-dynparam
                (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
                (cons "label_value" (position target-obj label-names :test #'string=)))
              (send *ri* :set-object-segmentation-candidates
                    :arm arm
                    :candidates (mapcar #'(lambda (x) (position x label-names :test #'string=))
                                        (append (list "background") bin-contents)))
              (if (send *ri* :check-bin-exist target-bin)
                (progn
                  (ros::set-param
                    (format nil "~a_hand/target_bin" (arm2str arm))
                    (symbol2str target-bin))
                  ;; logging
                  (ros::ros-info-blue "[~a] [main] state: ~a, target-bin: ~a, target-obj: ~a" (ros::get-name) state target-bin target-obj)
                  (setq state :pick_object)
                  )
                (progn
                  (ros::ros-warn "[~a] [main] could not find bin box: ~a" (ros::get-name) target-bin)
                  (setq state :set_target)
                  )
                )
              )
            )
          )
        (:wait_for_opposite_arm
          (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
          (ros::ros-info-blue "[~a] [main] ~a, ~a" (ros::get-name) arm state)
          (while
            (send *ri* :need-to-wait-opposite-arm arm)
            (unix::sleep 1))
          (setq state :pick_object)
          )
        (:pick_object
          (ros::ros-info-blue "[~a] [main] ~a, ~a" (ros::get-name) arm state)
          ;; if necessary wait for opposite arm
          (unix::sleep 2)
          (if (send *ri* :need-to-wait-opposite-arm arm)
            (progn
              (ros::ros-info-green "[~a] [main] Need to wait for opposite arm: ~a ~a" (ros::get-name) arm target-bin)
              (setq state :wait_for_opposite_arm)
              )
            (progn
              ;; set state to pick here
              (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
              ;; run motion
              (ros::ros-info-blue "[~a] [main] Recognizing objects in bin ~a" (ros::get-name) target-bin)
              (send *ri* :move-arm-body->bin-overlook-pose arm target-bin)
              (send *ri* :wait-interpolation)
              (if (send *ri* :recognize-objects-in-bin-loop arm target-bin)
                (progn
                  (setq object-found-p t)
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
                              :target-offset (if (eq arm :rarm) #f(10 0 0) #f(50 0 0))
                              )
                        )
                  (pushback graspingp past-graspingps)
                  )
                (progn
                  (setq graspingp nil)
                  (setq object-found-p nil)
                  )
                )
              (setq state :verify_object)
              )
            ))
        (:verify_object
          (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
          (ros::ros-info-blue "[~a] [main] ~a, ~a" (ros::get-name) arm state)
          (ros::ros-info-blue "[~a] [main] arm: ~a graspingp: ~a" (ros::get-name) arm graspingp)
          ;; Avoid shelf for safety
          (send *ri* :angle-vector-sequence
                (list (send *baxter* :avoid-shelf-pose arm target-bin))
                :fast :no-gripper-controller 0 :scale 5.0)
          (send *ri* :wait-interpolation)
          ;; Skip verification if no object is grasped
          (if graspingp
            ;; TODO(unknown): Make verification more stable, currently we skip the verification.
            ;; (if (or (= (length (send *ri* :get-bin-contents target-bin)) 1)
            ;;         ;; write here to skip verification like (list :a :b :c)
            ;;         (find target-bin (list nil)))
            ;; TODO: Make verification motion for gripper-v5
            (if t
              (progn (setq state :place_object))
              (progn
                (send *baxter* :view-hand-pose arm target-bin)
                (setq avs-picked->view-pose (send *ri* :send-av 5000))
                (send *ri* :wait-interpolation)
                (setq verify-ret (send *ri* :verify-object arm target-obj))
                (ros::ros-info-blue "[~a] [main] arm: ~a, verify-ret: ~a" (ros::get-name) arm verify-ret)
                (cond
                  ((eq verify-ret :timeout) (setq state :return_object))
                  ((null verify-ret) (setq state :return_object))
                  (verify-ret (setq state :place_object))
                  (t (setq state :return_object))
                  )
                )
              )
            (progn
              (send *ri* :stop-grasp arm)
              (if (eq arm :rarm)
                ;; gripper-v5
                (progn
                  (send *ri* :angle-vector-sequence
                        (list (send *baxter* :fold-to-keep-object arm))
                        :fast nil 0 :scale 5.0)
                  (send *ri* :wait-interpolation)
                  ;; open fingers
                  (send *ri* :move-hand arm
                        (send *baxter* :hand-grasp-pre-pose arm :cylindrical) 1500)
                  (send *ri* :move-hand arm
                        (send *baxter* :hand-grasp-pre-pose arm :opposed) 1500)
                  (send *ri* :angle-vector-sequence
                        (list (send *baxter* :fold-pose-back arm))
                        :fast nil 0 :scale 5.0)
                  (send *ri* :wait-interpolation)
                  )
                (send *ri* :angle-vector-sequence
                      (list
                        (send *baxter* :fold-to-keep-object arm)
                        (send *baxter* :fold-pose-back arm)
                        )
                      :fast nil 0 :scale 5.0)
                )
              ;; if have tried N times and N is larger than
              ;; number of bin contents, robot should abort the target
              (if (send *ri* :keep-picking-p target-bin past-graspingps object-found-p)
                (setq state :pick_object)
                (setq state :set_target)
                )
              )
            )
          )
        (:return_object
          ;; TODO: Make returning motion for gripper-v5
          (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
          (ros::ros-info-blue "[main] ~a, ~a" arm state)
          (send *ri* :angle-vector-sequence (reverse avs-picked->view-pose) :fast nil 0 :scale 5.0)
          (send *ri* :wait-interpolation)
          (case target-bin
            ((:a :b :c :j :k :l) (setq offset-avoid-bin-top 40))
            ((:d :e :f :g :h :i) (setq offset-avoid-bin-top 30)))
          (send *baxter* :rotate-gripper arm 45 :relative nil)
          (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
          (setq avs-picked->return-bin
                (cons
                  (progn
                    (send *baxter* :avoid-shelf-pose arm target-bin)
                    (send *baxter* :rotate-gripper arm 45 :relative nil))
                  (mapcar
                    #'(lambda (x)
                        (send *ri* :ik->bin-entrance arm target-bin
                              :offset (float-vector x 0 offset-avoid-bin-top)
                              :gripper-angle 45))
                    '(-100 -50 0 150 180))
                  )
                )
          (send *ri* :angle-vector-sequence avs-picked->return-bin :fast nil 0 :scale 5.0)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm) ;; release object
          (send *ri* :spin-off-by-wrist arm :times 20)
          (send *ri* :wait-interpolation)
          (setq avs-returned->exit-bin
                (mapcar
                  #'(lambda (x)
                      (send *ri* :ik->bin-entrance arm target-bin
                            :offset (float-vector x 0 offset-avoid-bin-top)
                            :gripper-angle 0))
                  '(180 150 0 -50 -100))
                )
          (pushback (send *baxter* :avoid-shelf-pose arm target-bin) avs-returned->exit-bin)
          (send *ri* :angle-vector-sequence
               avs-returned->exit-bin
                :fast nil 0 :scale 5.0)
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector (send *baxter* :avoid-shelf-pose arm target-bin))
          (send *ri* :wait-interpolation)
          (send *ri* :fold-pose-back arm)
          (send *ri* :wait-interpolation)
          ;; if have tried N times and N is larger than
          ;; number of bin contents, robot should abort the target
          (if (send *ri* :keep-picking-p target-bin past-graspingps object-found-p)
            (setq state :pick_object)
            (setq state :set_target)
            (ros::ros-warn "[main] Abort ~a and go to next target" target-bin))
          )
        (:place_object
          (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state))
          (ros::ros-info-blue "[main] ~a, ~a" arm state)
          (send *ri* :gripper-servo-off arm)  ;; power off gripper servo not to overload it
          (send *ri* :move-arm-body->order-bin arm :no-gripper-controller)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm)
          (if (eq arm :rarm)
            (progn
              ;; gripper-v5
              (send *ri* :move-hand arm
                    (send *baxter* :hand-grasp-pre-pose arm (send *ri* :get-val '_grasp-style))
                    1500)
              (send *baxter* :slide-gripper arm 100 :relative nil)
              (send *ri* :send-av 1000)
              (send *ri* :wait-interpolation)
              (send *baxter* :slide-gripper arm 0 :relative nil)
              (send *ri* :send-av 1000)
              (send *ri* :wait-interpolation)
              (send *ri* :move-hand arm
                    (send *baxter* :hand-grasp-pre-pose arm :opposed) 1500)
              (send *ri* :spin-off-by-wrist arm :times 5)
              )
            (send *ri* :spin-off-by-wrist arm :times 20)
            )
          (send *ri* :gripper-servo-on arm)
          (send *ri* :wait-interpolation)
          (send *baxter* :fold-pose-back arm)
          (send *ri* :send-av)
          (send *ri* :wait-interpolation)
          (setq state :set_target)
          )
        ))  ;; while & case
    ))

(warn "~% Commands ~%")
(warn "(apc-init)     : initialize *ri*~%")
(warn "(apc-mainloop) : start the mainloop~%~%")
