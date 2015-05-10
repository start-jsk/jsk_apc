;; -*- mode: lisp;-*-

(ros::load-ros-manifest "std_msgs")

(defun tweet (str)
  (unless (ros::get-topic-publisher "/tweet")
    (ros::advertise "/tweet" std_msgs::String 1)
    (unix:sleep 1))
  (let ((msg (instance std_msgs::string :init)))
    (send msg :data str)
    (ros::publish "/tweet" msg)
    (ros::spin-once)
    ))
