; (ros::load-ros-manifest "roseus")
; (ros::load-ros-manifest "std_msgs")
;
; (defparameter correct? nil)
(defparameter correct? t)
;
; (defun bool-cb (msg) (setq correct? (send msg :data)))
; (ros::subscribe "/object_verification" std_msgs::bool #'bool-cb)

(defun verify-object ()
  ; (ros::spin-once)
  correct?)