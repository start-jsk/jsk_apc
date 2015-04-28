;;
;; Check if grabbed or not
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "roseus")
(ros::roseus "grabbed")

(defparameter *grabbed-left* nil)
(defparameter *grabbed-right* nil)

(defun grabbed-left-cb (msg)
  (ros::ros-info (format nil "~A" (send msg :data)))
  (setq *grabbed-left* (send msg :data)))

(defun grabbed-right-cb (msg)
  (ros::ros-info (format nil "~A" (send msg :data)))
  (setq *grabbed-right* (send msg :data)))

(ros::subscribe "gripper_grabbed/limb/left/state" std_msgs::Bool #'grabbed-left-cb)
(ros::subscribe "gripper_grabbed/limb/right/state" std_msgs::Bool #'grabbed-right-cb)

(defun check-if-grabbed (l/r)
  (let (state)
    (case l/r
      (:larm
       (setq state *grabbed-left*))
      (:rarm
       (setq state *grabbed-right*)))
    state)
