#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")
(require "package://baxtereus/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-init.l")

(apc-init)
(send *baxter* :reset-pose)
(send *ri* :angle-vector (send *baxter* :angle-vector) 5000)
(send *ri* :wait-interpolation)
(send *baxter* :angle-vector #f(-2.32 -15.82 -38.93 -28.7183 74.6631 0.98877 -43.2202 -36.3647 15.82 -38.93 28.7183 74.6631 -0.98877 -43.2202 36.3647))
;; $ (send *baxter* :larm :end-coords :worldpos)
;; #f(1271.78 486.012 1413.35)
;; $ (rpy-angle (send *baxter* :larm :end-coords :worldrot))
;; ((0.24047 -0.18562 -1.00988) (3.38206 -2.95597 2.13171))
(send *baxter* :larm :inverse-kinematics (make-coords :pos #f(1271.78 486.012 1413.35) :rpy #f(0.24047 -0.18562 -1.00988)))
(send *baxter* :rarm :inverse-kinematics (make-coords :pos #f(1271.78 -486.012 1413.35) :rpy #f(-0.24047 -0.18562 1.00988)))
;;(send *ri* :angle-vector #f(-2.32 -15.82 -38.93 -28.7183 74.6631 0.98877 -43.2202 -36.3647 15.82 -38.93 28.7183 74.6631 -0.98877 -43.2202 36.3647) 5000)
(send *ri* :angle-vector (send *baxter* :angle-vector) 5000)
(send *ri* :wait-interpolation)
(exit)
