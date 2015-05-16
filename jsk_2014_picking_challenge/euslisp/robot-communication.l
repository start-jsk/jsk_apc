;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "std_msgs")

(defun tweet (str)
  (ros::advertise-if-yet "/tweet" std_msgs::String 1)
  (let ((msg (instance std_msgs::string :init)))
    (send msg :data str)
    (ros::publish "/tweet" msg)
    (ros::spin-once)
    ))

;; this requires iamge_saver.launch
(defun tweet-with-image (str l/r) ;; l/r is "larm" or "rarm"
  (ros::advertise-if-yet "/tweet" std_msgs::String 1)
  (let ((timestamp0 0) (timestamp1 0) (filename (format nil "/tmp/baxter_~A_camera.png" l/r)))
    (if (probe-file filename)
      (setq timestamp0 (file-write-date filename)))
    (setq timestamp1 timestamp0)
    (call-empty-service (format nil "/~A_image_saver/save" l/r))
    (setq tm (ros::time-now))
    (ros::ros-info "~A timestamp ~A ~A" filename timestamp0 timestamp1)
    (ros::rate 2)
    (while (and (< (send (ros::time- (ros::time-now) tm) :to-sec) 10)
                (= timestamp0 timestamp1))
           (if (probe-file filename) (setq timestamp1 (file-write-date filename)))
           (ros::ros-info "~A timestamp ~A ~A" filename timestamp0 timestamp1)
           (ros::sleep))
    (ros::ros-info "save image")
    (tweet (format nil "~A ~A" str filename))
    ))

