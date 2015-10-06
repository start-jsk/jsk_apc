#!/usr/bin/env roseus

(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")
(ros::load-ros-manifest "std_msgs")

(defun tweet (str)
  (let ((msg (instance std_msgs::string :init)))
    (send msg :data str)
    (ros::publish "/tweet" msg)
    (ros::spin-once)
    ))

;; this requires iamge_saver.launch
(defun tweet-with-image (str l/r) ;; l/r is "left" or "right"
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
    (tweet (format nil "~A ~A" (subseq str 0 100) filename)) ;; some bugs on rostwitter did not publish with long data
    ))

(ros::roseus "tweet_server")
(ros::advertise-if-yet "/tweet" std_msgs::String 1)
(ros::ros-info "start advertising /tweet")
(ros::subscribe "/tweet_larm" std_msgs::String #'(lambda (x) (tweet-with-image (send x :data) "left")))
(ros::subscribe "/tweet_rarm" std_msgs::String #'(lambda (x) (tweet-with-image (send x :data) "right")))
(ros::subscribe "/tweet_string" std_msgs::String #'(lambda (x) (tweet (send x :data))))
(ros::ros-info "start subscribing /tweet_larm")
(ros::ros-info "start subscribing /tweet_rarm")
(ros::ros-info "start subscribing /tweet_string")
(ros::spin)

