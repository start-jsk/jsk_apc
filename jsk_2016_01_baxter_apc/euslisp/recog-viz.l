#!/usr/bin/env roseus

#|
;;
;; imageview2 sample program
;;
;; before invoke this script, first launch following programs
;;

(ros::roseus-add-msgs "geometry_msgs")
(ros::roseus-add-msgs "image_view2")
;;;
;;;
(ros::roseus "imageview2-client")

(ros::advertise "image_marker" image_view2::ImageMarker2 1)

;;(ros::subscribe "/color_object_matcher/output" jsk_recognition_msgs::ClassificationResult #'print-recog-cb)

(setq label-names "hello world")
(ros::rate 10)
(while (ros::ok)
    (setq mrk (instance image_view2::ImageMarker2 :init))
       (send mrk :type image_view2::ImageMarker2::*TEXT*)
         (send mrk :position (instance geometry_msgs::Point :init :x 320 :y 240))
	 (send mrk :scale 0.5)
	 (send mrk :text label-names))
    (ros::publish "image_marker" mrk)
    (ros::ros-info "type ~A" label-names)
	(ros::spin-once)
	(ros::sleep))


(ros::spin) 
