#!/usr/bin/env roseus

(ros::load-ros-manifest "jsk_2014_picking_challenge")
(ros::load-ros-manifest "jsk_rviz_plugins")

(ros::roseus "apc_tweet")
(ros::advertise "tweet" std_msgs::string 1)
(ros::subscribe "/semi/master_status" jsk_rviz_plugins::OverlayText 
		#'(lambda (msg)
		    (ros::ros-info (format nil "I heard ~A" (send msg :text)))
		    (ros::publish "tweet" std_msgs::string (send msg :text))
		    (tweet-string (send msg :text)
				  :warning-time nil
				  :with-image "/cameras/head_camera/image")
		    ))

(ros::spin)







