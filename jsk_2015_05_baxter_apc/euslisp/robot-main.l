#!/usr/bin/env roseus
;;
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")

(ros::roseus "robot_main")

(apc-init)

(defun main ()
  (move-to-target-bin :rarm :c)
  ; (pick-object :rarm)
  (move-for-verification :rarm)
  (place-object :rarm))

