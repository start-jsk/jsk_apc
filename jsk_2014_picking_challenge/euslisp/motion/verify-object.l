(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")

(defparameter correct? nil)

(defun bool-cb (msg) (setq correct? (send msg :data)))
(ros::subscribe "/object_verification/output" std_msgs::bool #'bool-cb)

(defun verify-object ()
  (ros::spin-once)
  correct?)