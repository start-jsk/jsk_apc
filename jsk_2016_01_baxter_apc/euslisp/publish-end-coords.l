;; vim: set ft=lisp:
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")

(jsk_2016_01_baxter_apc::baxter-init)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(ros::advertise "/rarm_end_coords" geometry_msgs::PoseStamped)
(unix::sleep 1)

(setq msg (instance geometry_msgs::PoseStamped :init))
(send msg :header :frame_id "base")
(send msg :pose (ros::coords->tf-pose (send *baxter* :rarm :end-coords :worldcoords)))
(send msg :header :stamp (ros::time-now))
(ros::publish "/rarm_end_coords" msg)
(exit)
