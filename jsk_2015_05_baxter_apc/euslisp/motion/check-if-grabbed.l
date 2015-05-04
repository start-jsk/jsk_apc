;;
;; Check if grabbed or not
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "roseus")

(defparameter *grabbed-left* nil)
(defparameter *grabbed-right* nil)

(ros::subscribe "gripper_grabbed/limb/left/state" std_msgs::Bool
                #'(lambda (msg) (setq *grabbed-left* (send msg :data))))
(ros::subscribe "gripper_grabbed/limb/right/state" std_msgs::Bool
                #'(lambda (msg) (setq *grabbed-right* (send msg :data))))

(defun check-if-grabbed (arm)
  (let (state)
    (ros::spin-once)
    (case arm
      (:larm
       (setq state *grabbed-left*))
      (:rarm
       (setq state *grabbed-right*)))
    (when state (speak-en "Grasped object" :google t))
    state)
