(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(setq correct? nil)
(setq verify-object-stamp nil)

(defun object-verification-cb (msg)
  (setq correct? (send msg :data))
  (setq verify-object-stamp (send (send msg :header) :stamp)))
(ros::subscribe "/object_verification/output" jsk_2014_picking_challenge::BoolStamped #'object-verification-cb)

(defun verify-object (&key (timeout 20))
  (setq now (ros::time-now))
  (while
    (< (send verify-object-stamp :sec) (send now :sec))
    (unix::usleep 1000)
    (ros::spin-once)
    (when (> (- (send (ros::time-now) :sec) (send now :sec)) timeout) (return)))
  correct?)