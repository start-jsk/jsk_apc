#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")
(load "package://baxtereus/baxter-interface.l")

(baxter-init)
(send *baxter* :reset-pose)
(send *ri* :angle-vector (send *baxter* :angle-vector) 5000)
(send *ri* :wait-interpolation)
(send *ri* :angle-vector #f(-2.32 -15.82 -38.93 -28.7183 74.6631 0.98877 -43.2202 -36.3647 15.82 -38.93 28.7183 74.6631 -0.98877 -43.2202 36.3647) 5000)
(send *ri* :wait-interpolation)
(exit)
