#!/usr/bin/env roseus
;;
(load "package://baxtereus/baxter-interface.l")

(ros::roseus "on_off_gripper")
(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(ros::advertise "on_off_gripper" std_msgs::Empty 1)

(while (ros::ok)
  (ros::rate 1)
  (setq msg (instance std_msgs::Empty :init))
  (ros::publish "on_off_gripper" msg)
  (ros::sleep))

;; (ros::roseus "shutdown")
;; (exit)
