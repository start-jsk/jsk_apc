#!/usr/bin/env roseus

;;; euslisp version of move_arm_publisher.l

(ros::load-ros-manifest "roseus")

(ros::roseus "euslisp_move_arm")   ;; node's name
(ros::advertise "semi/move_arm_chatter" geometry_msgs::PoseStamped)  ;; topic's name and massage's type

(while (ros::ok)
  (setq msg (instance geometry_msgs::PoseStamped :init))
  (send msg :header :frame_id "/base")
  (send msg :header :stamp (ros::time-now))
  (send msg :pose :position :x 0.97)
  (send msg :pose :position :y -0.40)
  (send msg :pose :position :z 0.45)
  (send msg :pose :orientation :x 0.55)
  (send msg :pose :orientation :y 0.80)
  (send msg :pose :orientation :z 0.09)
  (send msg :pose :orientation :w 0.59)
  (ros::publish "semi/move_arm_chatter" msg)
  (ros::sleep)
)

(ros::roseus "shutdown")
(exit)