#!/usr/bin/env roseus
;;
;; To use this, you have to run below on
;; **PC to which Arduino is connected**
;; to control vacuum cleaner.
;;
;; $ rosrun rosserial_python serial_node.py _port:=/dev/ttyUSB0
;;
(ros::load-ros-manifest "roseus")
(ros::roseus "control_vacuum_gripper")
(ros::advertise "on_off_gripper" std_msgs::Bool 1)
(ros::rate 10)

(defmethod baxter-interface
  (:start-grasp
    ()
    (setq msg (instance std_msgs::Bool :init))
    (send msg :data t)
    (ros::publish "on_off_gripper" msg)
    (ros::sleep))
  (:go-grasp () (send self :start-grasp))
  (:stop-grasp
    ()
    (setq msg (instance std_msgs::Bool :init))
    (send msg :data nil)
    (ros::publish "on_off_gripper" msg)
    (ros::sleep))
  )
