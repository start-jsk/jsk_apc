#!/usr/bin/env roseus
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(ros::roseus "robot_main")

;;Global Variables

;; end of Global Variables

(setq client (instance ros::simple-action-client :init
		  "/picking_server/rarm" jsk_2014_picking_challenge::EusCommandAction))
(ros::ros-info "waiting ~A available" "picking_server/rarm")
(send client :wait-for-server)

(setq goal (instance jsk_2014_picking_challenge::EusCommandActionGoal :init))
(send goal :header :stamp (ros::time-now))
(send goal :goal_id :stamp (ros::time-now))
(send goal :goal :command  "move_to_target")
(send goal :goal :target_bin "c")
(send client :send-goal goal))

(send client :wait-for-result)
