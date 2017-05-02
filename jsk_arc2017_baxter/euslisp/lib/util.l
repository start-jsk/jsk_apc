#!/usr/bin/env roseus

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")

(ros::load-ros-manifest "jsk_gui_msgs")
(ros::load-ros-manifest "jsk_recognition_msgs")

(defun opposite-arm (arm) (if (eq arm :larm) :rarm :larm))

(defun get-arm-controller (arm)
  (cond ((eq arm :rarm) :rarm-controller)
        ((eq arm :larm) :larm-controller)
        (t nil)))

(defun get-bin-contents (bin)
  (ros::get-param (format nil "/bin_contents/~A"
                            (string-upcase (symbol2str bin)))))
