;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "std_msgs")

(defun tweet (str)
  (ros::advertise-if-yet "/tweet" std_msgs::String 1)
  (let ((msg (instance std_msgs::string :init)))
    (send msg :data str)
    (ros::publish "/tweet" msg)
    (ros::spin-once)
    ))
