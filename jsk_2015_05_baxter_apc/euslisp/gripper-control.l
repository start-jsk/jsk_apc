#!/usr/bin/env roseus
;;
;; To use this, you have to run below on
;; **PC to which Arduino is connected**
;; to control vacuum cleaner.
;;
;; $ rosrun rosserial_python serial_node.py _port:=/dev/ttyUSB0
;;
(ros::load-ros-manifest "roseus")
(ros::roseus "start_grasp_override")
(ros::advertise "on_off_gripper" std_msgs::Bool 1)
(ros::rate 10)

(defmethod baxter-interface
  (:go-grasp
    ()
    (ros::ros-info (format nil "[on_off_gripper] cleaner ON"))
    (setq msg (instance std_msgs::Bool :init))
    (send msg :data t)
    (ros::publish "on_off_gripper" msg)
    (ros::sleep))
  (:stop-grasp
    ()
    (ros::ros-info (format nil "[on_off_gripper] cleaner OFF"))
    (setq msg (instance std_msgs::Bool :init))
    (send msg :data nil)
    (ros::publish "on_off_gripper" msg)
    (ros::sleep))
  )
