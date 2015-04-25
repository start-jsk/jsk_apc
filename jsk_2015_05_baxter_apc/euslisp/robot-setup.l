#!/usr/bin/env roseus
;;
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")

(ros::roseus "robot_setup")

(defun setup ()
  ; this should be called once by human
  (ros::set-param "/target" "a")
  (ros::set-param "/left_limb/state" "move_to_target_bin")
  (ros::set-param "/right_limb/state" "move_to_target_bin")
  (send *ri* :angle-vector-sequence (append (fold-pose-back :larm) (fold-pose-back :rarm)))
  )

(apc-init)
(setup)
(exit)
