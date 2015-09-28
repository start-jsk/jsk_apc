#!/usr/bin/env roseus
;;
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/target-pick-region.l")

(ros::roseus "move_arm2target_bin_listener")
(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")

(defun init ()
  ;; baxter model
  (baxter-init)
  (send *baxter* :locate #f(0 0 950) :world)
  ;; pod-lowres model
  (when (not (boundp '*pod*))
    (load "package://jsk_2015_05_baxter_apc/euslisp/pod-lowres.l")
    (send *pod* :locate #f(1300 0 0) :world)
    (send *pod* :rotate pi/2 :x :world)
    (send *pod* :rotate (- pi/2) :z :world))
  (objects (list *baxter* *pod*))
  ;; subscriber init
  (ros::subscribe "/semi/move_larm2target_bin" std_msgs::String #'cb-larm)
  (ros::subscribe "/semi/move_rarm2target_bin" std_msgs::String #'cb-rarm)
  (ros::ros-info "subscriber initialized"))

(defun cb-larm (msg)
  (callback msg ':larm))

(defun cb-rarm (msg)
  (callback msg ':rarm))

(defun callback (msg arm)
  """Move arm to target region which is ordered with msg.
  msg data should be string and included in (a, b, c, ..., l)."""
  (let ((target-nm (intern (string-upcase (send msg :data))))
        (region (instance target-pick-region :init)) c)
    (ros::ros-info (format nil "moving to ~A." target-nm))
    ;; target data validation
    (unless (find target-nm (send region :target-name-list))
      (ros::ros-info (format nil "~A not found." target-nm))
      (return-from callback nil))
    ;; get target coordinates
    (setq c (send *pod* :transform-vector (send region :get-val target-nm)))
    ;; ik with z axis constraction
    (ros::ros-info (format nil "computing IK for ~A." target-nm))
    (send *baxter* arm
          :inverse-kinematics (instance cascaded-coords :init :pos c)
          :rotate-axis :z)
    ;; reflect change to ri
    (send *ri* :angle-vector (send *baxter* :angle-vector))
    (send *ri* :wait-interpolation)
    (send *irtviewer* :draw-objects)
    (x::window-main-one)
    ))

(init)
(ros::spin)

