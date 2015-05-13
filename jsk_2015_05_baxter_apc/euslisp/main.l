#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-input.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-visualization.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-recognition.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-communication.l")

(defun main (arm)
  (let (target param-target state work-order work)
    (setq work-order (get-work-order arm))
    ; get previous target from parameter server
    ; this does work when this robot_main node respawned
    (while (not (ros::get-param (format nil "~A_limb/target" (arm-to-str arm)))) (unix::sleep 1))
    (setq param-target (str-to-key (ros::get-param (format nil "~A_limb/target" (arm-to-str arm)))))
    ; remove already picked target bins
    ; this does work when this robot_main node respawned
    (while
      t
      (setq work (pop work-order))
      (when (eq (elt work 0) param-target) (return)))
    ; loop of motions
    (setq n-tried 0)
    (speak-en "Let's start picking challenge" :google t)
    (while
      work
      ; get status from parameter server
      ;; (speak-en  (format nil "Next Target is ~A" "OREO") :google t  :wait t)
      (setq target (str-to-key (elt work 0)) target-object (elt work 1))
      (ros::set-param "~A_limb/target" (key-to-str target))
      (setq state (ros::get-param (format nil "~A_limb/state" (arm-to-str arm))))
      (ros::ros-info "state: ~A, arm: ~A, target: ~A, object: ~A"
                     state (arm-to-str arm) (key-to-str target) target-object)
      (cond
        ((string= state "pick_object")
         (incf n-tried)
         (ros::ros-info "Move to Bin ~A. Target is ~A." (key-to-str target) target-object)
         (speak-en (format nil "Move to Bin ~A. Target is ~A." (key-to-str target) (underscore-to-space target-object)) :google t)
         (move-to-target-bin arm target)
         (speak-en (format nil "Pick Object in ~A." (key-to-str target)) :google t)
         (pick-object arm target)
         (cond ((= (length (get-bin-contents target)) 1) ;; SINGLE-ITEM-BIN
                (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "place_object"))
               ((= (- (length (get-bin-contents target)) n-tried) 0)  ;; already tried enough
                (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "place_object"))
               ((not (check-if-grabbed arm))  ;; this is temporary for uncertain check-if-grabbed
                (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "pick_object"))
               (t (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "move_for_verification"))))
        ((string= state "move_for_verification")
         (speak-en  "Move for verification." :google t)
         (move-for-verification arm target)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "verify_object"))
        ((string= state "verify_object")
         (let ((correct? (verify-object arm target-object)))
           (if correct?
             (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "place_object")
             (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "return_object"))))
        ((string= state "return_object")
         (speak-en (format nil "Return object to Bin ~A." (key-to-str target)) :google t)
         (move-to-target-bin arm target)
         (return-object arm target)
         ;; if have tried N times and N is larger than number of bin contents,
         ;; robot should abort the target
         (when (<= (length (get-bin-contents target)) n-tried)
           (setq work (pop work-order) n-tried 0))
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "pick_object"))
        ((string= state "place_object")
         (speak-en  "Target is verified. Put in Order bin." :google t)
         (place-object arm)
         (setq work (pop work-order) n-tried 0)  ;; initialize for loop
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "pick_object"))
        (t (ros::set-param (format nil "/~A_limb/state" (arm-to-str arm)) "pick_object"))
        )
      )
    ))

(warn "~% Commands ~%")
(warn "(apc-init) : Setup~%")
(warn "(main)     : Start the loop~%")
