#!/usr/bin/env roseus
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")

(ros::roseus "robot_main")

(defun main ()
  (let (arm target param-target state work)
    ; get previous target from parameter server
    ; this does work when this robot_main node respawned
    (setq param-target (str-to-symbol (ros::get-param "/target")))
    ; remove already picked target bins
    ; this does work when this robot_main node respawned
    (while
      t
      (setq work (pop workorder))
      (when (string= (when (elt work 1)) (symbol-to-str param-target))) (return))
    ; loop of motions
    (while
      work
      ; get status from parameter server
      (setq arm (str-to-arm (elt work 0)) target (str-to-symbol (elt work 1)))
      (setq state (ros::get-param (format nil "~A_limb/state" (arm-to-str arm))))
      (ros::ros-info "state: ~A, arm: ~A, target: ~A" state (arm-to-str arm) (symbol-to-str target))
      (cond
        ((string= state "move_to_target_bin")
         (move-to-target-bin arm target)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "pick_object"))
        ((string= state "pick_object")
         (pick-object arm)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "move_for_verification"))
        ((string= state "move_for_verification")
         (move-for-verification arm)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "place_object"))
        ((string= state "place_object")
         (setq arm (str-to-arm (elt work 0)) target (str-to-symbol (elt work 1)))
         (place-object arm)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "move_to_target_bin")
	  ; update arm and target
         (setq work (pop workorder))
	 (if work
	     (progn
	       (setq arm (str-to-arm (elt work 0)) target (str-to-symbol (elt work 1)))
	       (ros::set-param "/target" (symbol-to-str target)))))
        (t (ros::set-param (format nil "/~A_limb/state" (arm-to-str arm)) "move_to_target_bin"))
        )
      )
    ))

; this is test workorder
; TODO: make workorder publisher
(setq workorder (list '("right" :c) '("left" :a) '("right" :b) '("left" :d) '("right" :e)))

(apc-init)
(main)
