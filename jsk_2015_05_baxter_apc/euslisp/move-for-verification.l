#!/usr/bin/env roseus
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/arm-motion.l")
(load "package://jsk_2014_picking_challenge/euslisp/common.l")

(ros::load-ros-manifest "jsk_rviz_plugins")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defun callback (req)
  (let ((arm (str2arm (send req :limb))) avs res)
    (setq res (send req :response))
    (pushback (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
    ;; validate arm
    (unless (validate-arm arm)
      (send res :succeeded nil)
      (return-from :execute-cb res))
    ;; move arm
    (setq avs (append avs (move-for-verification arm)))
    (send *irtviewer* :draw-objects)
    (send res :succeeded t)
    res))

(ros::roseus "move_for_verification")
(model-init)
(ros::advertise-service "move_for_verification" jsk_2014_picking_challenge::MoveArm #'callback)
(ros::spin)
