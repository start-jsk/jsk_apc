#!/usr/bin/env roseus

(ros::roseus-add-msgs "geometry_msgs")
(ros::roseus-add-msgs "image_view2")
(ros::roseus-add-msgs "jsk_recognition_msgs")

(ros::roseus "viz-recog")
(ros::advertise "image_marker" image_view2::ImageMarker2 1)

(setq *label-names* nil)
(setq *color-list* (list "red" "blue" "yellow" "pink"))

(defun viz-recog-cb (msg)
  (setq *label-names* (send msg :label_names))
  (ros::ros-info "~A" *label-names*)
  )

(defun list-to-string (label-list color-list text)
    (if label-list
      (list-to-string
        (cdr label-list)
        (cdr color-list)
        (concatenate string text (car color-list) ":" (car label-list) "  "))
      text)
  )

(ros::subscribe
  "/color_object_matcher/output"
  jsk_recognition_msgs::ClassificationResult
  #'viz-recog-cb)

(ros::rate 10)
(setq i 0)
(while (ros::ok)
       (setq mrk (instance image_view2::ImageMarker2 :init))
       (send mrk :type image_view2::ImageMarker2::*TEXT*)
       (send mrk :position (instance geometry_msgs::Point :init :x 1080 :y 240))
       (send mrk :scale 1.2)
       (send mrk :text (list-to-string *label-names* *color-list* ""))
       (ros::publish "image_marker" mrk)
       (ros::ros-info "~A" (send mrk :text))
       (ros::spin-once)
       (ros::sleep))
