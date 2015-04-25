;;
;; Check if grabbed or not
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "roseus")
(ros::roseus "grabbed")

(defparameter *grabbed* nil)

(defun grabbed-cb (msg)
  (ros::ros-info (format nil "~A" (send msg :data)))
  (setq *grabbed* (send msg :data)))

(ros::subscribe "gripper_grabbed/state" std_msgs::Bool #'grabbed-cb)
