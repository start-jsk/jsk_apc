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
(ros::advertise "/on_off_gripper/limb/left" std_msgs::Bool 1)
(ros::advertise "/on_off_gripper/limb/right" std_msgs::Bool 1)
(ros::rate 10)

(defmethod baxter-interface
  (:start-grasp
   (&optional (arm :arms))
   (case arm
     (:larm
      (setq arm "left"))
     (:rarm
      (setq arm "right")))
   (ros::ros-info (format nil "~A" arm))
   (dolist (a (if (eq arm :arms) (list "left" "right") (list arm)))
     (setq msg (instance std_msgs::Bool :init))
     (send msg :data t)
     (ros::publish (concatenate string "/on_off_gripper/limb/" a) msg)
     (ros::ros-info (format nil "~A" (concatenate string "/on_off_gripper/limb/" a)))
     (ros::sleep)
     ))
  (:go-grasp () (send self :start-grasp))
  (:stop-grasp
   (&optional (arm :arms))
   (case arm
     (:larm
      (setq arm "left"))
     (:rarm
      (setq arm "right")))
   (ros::ros-info (format nil "~A" arm))
   (dolist (a (if (eq arm :arms) (list "left" "right") (list arm)))
     (setq msg (instance std_msgs::Bool :init))
     (send msg :data nil)
     (ros::ros-info (format nil "~A" (concatenate string "/on_off_gripper/limb/" a)))
     (ros::publish (concatenate string "/on_off_gripper/limb/" a) msg)
     (ros::sleep)
     ))
  )
