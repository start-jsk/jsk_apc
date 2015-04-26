#!/usr/bin/env roseus
;;

(ros::roseus "robot_main")

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-test.l")


(defun main ()
  (let (arm target param-target state work)
    ; get previous target from parameter server
    ; this does work when this robot_main node respawned
    (while (not (ros::get-param "/target")) (unix::sleep 1))
    (setq param-target (str-to-symbol (ros::get-param "/target")))
    ; remove already picked target bins
    ; this does work when this robot_main node respawned
    (while
      t
      (setq work (pop work-order))
      (when (string= (when (elt work 1)) (symbol-to-str param-target))) (return))
    ; loop of motions
    (speak-en "Let's start picking challenge" :google t :wait t)
    (while
      work
      ; get status from parameter server
      ;; (speak-en  (format nil "Next Target is ~A" "OREO") :google t  :wait t)
      (setq arm (str-to-arm (elt work 0)) target (str-to-symbol (elt work 1)))
      (setq state (ros::get-param (format nil "~A_limb/state" (arm-to-str arm))))
      (send *irtviewer* :draw-objects)
      (ros::ros-info "state: ~A, arm: ~A, target: ~A" state (arm-to-str arm) (symbol-to-str target))
      (cond
        ((string= state "move_to_target_bin")
	 (speak-en (format nil "Move to Bin ~A . Target is ~A" (symbol-to-str target) "OREO" ) :google t)
         (move-to-target-bin arm target)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "pick_object"))
        ((string= state "pick_object")
	 (speak-en (format nil "Pick Object in ~A." (symbol-to-str target)) :google t)
         (pick-object arm)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "move_for_verification"))
        ((string= state "move_for_verification")
	 (speak-en  "Move to verification." :google t)
         (move-for-verification arm)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "place_object"))
        ((string= state "place_object")
         (setq arm (str-to-arm (elt work 0)) target (str-to-symbol (elt work 1)))
	 (speak-en  "Target is verified. Put in Order bin" :google t)
         (place-object arm)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "move_to_target_bin")
	  ; update arm and target
         (setq work (pop work-order))
	 (if work
	     (progn
	       (setq arm (str-to-arm (elt work 0)) target (str-to-symbol (elt work 1)))
	       (ros::set-param "/target" (symbol-to-str target)))))
        (t (ros::set-param (format nil "/~A_limb/state" (arm-to-str arm)) "move_to_target_bin"))
        )
      )
    ))

(setq work-order (get-work-order))

(warn "~% Commands ~%")
(warn "(apc-init) : Setup~%")
(warn "(main)     : Start the loop~%")
