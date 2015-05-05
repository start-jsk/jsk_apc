(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(setq correct-left? nil)
(setq correct-right? nil)
(setq verify-object-stamp-left nil)
(setq verify-object-stamp-right nil)

(defun object-verification-cb (msg l/r)
  (case l/r
    (:left
      (setq correct-left? (send msg :data))
      (setq verify-object-stamp-left (send (send msg :header) :stamp)))
    (:right
      (setq correct-right? (send msg :data))
      (setq verify-object-stamp-right (send (send msg :header) :stamp)))
    ))
(ros::subscribe "/left_process/object_verification/output" jsk_2014_picking_challenge::BoolStamped
                #'(lambda (msg) (object-verification-cb msg :left)))
(ros::subscribe "/right_process/object_verification/output" jsk_2014_picking_challenge::BoolStamped
                #'(lambda (msg) (object-verification-cb msg :right)))

(defun verify-object (arm)
  (let (veify-object-stamp (now (ros::time-now)))
    (case arm
      (:larm (setq verify-object-stamp verify-object-stamp-left))
      (:rarm (setq verify-object-stamp verify-object-stamp-right)))
    (while  ;; wait for msg
      (< (send verify-object-stamp :sec) (send now :sec))
      (unix::usleep 1000)
      (ros::spin-once))
    (case arm
      (:larm correct-left?)
      (:rarm correct-right?))
    ))