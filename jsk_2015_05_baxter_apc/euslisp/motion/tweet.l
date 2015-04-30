(ros::load-ros-manifest "jsk_2014_picking_challenge")

(ros::advertise "/tweet" std_msgs::string 1)
(defun tweet (str)
  (let ((msg (instance std_msgs::string :init)))
    (send msg :data str)
    (ros::publish "tweet" msg)
    (ros::spin-once)
    ))