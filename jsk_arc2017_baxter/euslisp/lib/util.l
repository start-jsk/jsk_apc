#!/usr/bin/env roseus

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")

(ros::load-ros-manifest "jsk_gui_msgs")
(ros::load-ros-manifest "jsk_recognition_msgs")

(defun opposite-arm (arm) (if (eq arm :larm) :rarm :larm))

(defun get-bin-contents (bin)
  (ros::get-param (format nil "/bin_contents/~A"
                            (string-upcase (symbol2str bin)))))

(defun wait-for-user-input-to-start (arm)
  (let (can-start)
    (ros::ros-info "[:wait-for-user-input-to-start] wait for user input to start: ~a" arm)
    (ros::wait-for-service "/rviz/yes_no_button")
    (while
      (not can-start)
      (setq can-start (send (ros::service-call
                              "/rviz/yes_no_button" (instance jsk_gui_msgs::YesNoRequest)) :yes)))
    (ros::ros-info "[:wait-for-user-input-to-start] received user input: ~a" arm)))

(defun set-object-segmentation-candidates (arm candidates)
  (let ((req (instance jsk_recognition_msgs::SetLabelsRequest :init)))
    (send req :labels candidates)
    (ros::service-call
      (format nil "/~a_hand_camera/apply_context_to_label_proba/update_candidates"
              (arm2str arm))
      req)))

(defun set-arm-state-param (arm state)
  (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state)))

(defun get-arm-controller (arm)
  (cond ((eq arm :rarm) :rarm-controller)
        ((eq arm :larm) :larm-controller)
        (t nil)))
