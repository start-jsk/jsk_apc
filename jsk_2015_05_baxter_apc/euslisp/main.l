#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2015_05_baxter_apc/euslisp/robot-recognition.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/model/pod-lowres.l")

(defun apc-init (&optional (ctype :default-controller))
  (jsk_2015_05_baxter_apc::baxter-init :ctype ctype)
  (objects (list *baxter*))
  t)

(defun main (arm)
  (bounding-box-array-in-bin :timeout 3)
  (while  ;; wait for user input to start
    (equal (ros::get-param (format nil "~A_process/state" (arm-to-str arm))) "wait_for_user_input")
    (ros::ros-info "waiting for user input to start: ~A" arm)
    (unix::sleep 1))
  (let (target param-target state work-order work)
    (setq work-order (send *ri* :get-work-order arm))
    (send *ri* :recognize-bin-boxes)

    ; get previous target from parameter server
    ; this does work when this robot_main node respawned
    (while (not (ros::get-param (format nil "~A_process/target" (arm-to-str arm)))) (unix::sleep 1))
    (setq param-target (str-to-key (ros::get-param (format nil "~A_process/target" (arm-to-str arm)))))
    ; remove already picked target bins
    ; this does work when this robot_main node respawned
    (while
      t
      (setq work (pop work-order))
      (when (eq (elt work 0) param-target) (return)))
    ; loop of motions
    (setq n-tried 0)  ;; how many tried to pick
    (speak-en "Let's start picking challenge" :google t)
    (while
      work
      ;; get status from parameter server
      ;; (speak-en  (format nil "Next Target is ~A" "OREO") :google t  :wait t)
      (setq target (str-to-key (elt work 0)) target-object (elt work 1))
      (ros::set-param (format nil "~A_process/target" (arm-to-str arm)) (key-to-str target))
      (setq state (ros::get-param (format nil "~A_process/state" (arm-to-str arm))))
      (ros::ros-info "state: ~A, arm: ~A, target: ~A, object: ~A"
                     state (arm-to-str arm) (key-to-str target) target-object)
      (cond
        ((string= state "pick_object")
         ;; wait for opposite arm if opposite arm's target is center bin
         (while
           (eq
             (which-bin-region (str-to-key (ros::get-param (format nil "~A_process/target" (arm-to-str (opposite-arm arm))))))
             :center)
           (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "wait_for_opposite")
           (ros::ros-info "waiting for opposite arm to finish one order for center bin")
           (move-to-wait-pose arm target)
           (unix::sleep 1))
         (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "pick_object")
         ;; wait for opposite arm if current arm's target is center bin and opposite is still processing
         (while
           (and
             (eq (which-bin-region target) :center)
             (not (string= (ros::get-param (format nil "~A_process/state" (arm-to-str (opposite-arm arm)))) "wait_for_opposite")))
           (ros::ros-info "waiting for opposite arm to do order for center bin")
           (unix::sleep 1))
         (ros::ros-info "Move to Bin ~A. Target is ~A." (key-to-str target) target-object)
         (speak-en (format nil "Move to Bin ~A. Target is ~A." (key-to-str target) (underscore-to-space target-object)) :google t)
         (send *ri* :move-arm-body->bin arm target)
         (send *ri* :wait-interpolation)
         (speak-en (format nil "Pick Object in ~A." (key-to-str target)) :google t)
         (send *ri* :pick-object arm target)
         (incf n-tried)  ;; how many times to try to pick
         (if (= (length (get-bin-contents target)) 1)
           (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "place_object")  ;; SINGLE-ITEM-BIN
           (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "move_for_verification")))  ;; MULTI-ITEM-BIN
        ((string= state "move_for_verification")
         (speak-en  "Move for verification." :google t)
         (send *ri* :move-arm-body->head-view-point arm)
         (send *ri* :wait-interpolation)
         (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "verify_object"))
        ((string= state "verify_object")
         (let ((correct? (send *ri* :verify-object arm target-object)))
           (if correct?
             (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "place_object")
             (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "return_object"))))
        ((string= state "return_object")
         (speak-en (format nil "Return object to Bin ~A." (key-to-str target)) :google t)
         (send *ri* :move-arm-body->bin arm target)
         (send *ri* :wait-interpolation)
         (send *ri* :angle-vector-sequence (send *ri* :ik-avs->bin-entrance arm target :offset #f(300 0 80)))
         (send *ri* :wait-interpolation)
         (send *ri* :stop-grasp)
         (send *ri* :spin-off-by-wrist arm)
         (send *ri* :wait-interpolation)
         ;; if have tried N times and N is larger than number of bin contents,
         ;; robot should abort the target
         (when (<= (length (get-bin-contents target)) n-tried)
           (setq work (pop work-order) n-tried 0))
         (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "pick_object"))
        ((string= state "place_object")
         (speak-en  "Target is verified. Put in Order bin." :google t)
         (send *ri* :move-arm-body->order-bin arm)
         (send *ri* :wait-interpolation)
         (send *ri* :spin-off-by-wrist arm)
         (send *ri* :wait-interpolation)

         (setq work (pop work-order) n-tried 0)  ;; initialize for loop
         (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "pick_object"))
        (t (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "pick_object"))
        )
      )
    (move-to-wait-pose arm target)
    (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "wait_for_opposite")
    (ros::set-param (format nil "~A_process/target" (arm-to-str arm)) "")
    ))

(warn "~% Commands ~%")
(warn "(apc-init) : Setup~%")
(warn "(main)     : Start the loop~%")
