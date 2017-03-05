#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")

(defun publish-end-caords (arm)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (setq msg (instance geometry_msgs::PoseStamped :init))
  (send msg :header :frame_id "base")
  (send msg :pose (ros::coords->tf-pose (send *baxter* arm :end-coords :worldcoords)))
  (send msg :header :stamp (ros::time-now))
  (if (eq arm :larm)
    (ros::publish "/larm_end_coords" msg)
    (ros::publish "/rarm_end_coords" msg)
    )
  )

(jsk_2016_01_baxter_apc::baxter-init)
(ros::advertise "/larm_end_coords" geometry_msgs::PoseStamped)
(ros::advertise "/rarm_end_coords" geometry_msgs::PoseStamped)
(unix::sleep 1)
(publish-end-caords :larm)
(publish-end-caords :rarm)
