#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l")

(ros::roseus "fold_pose_back")

(jsk_2015_05_baxter_apc::baxter-init)
(send *ri* :fold-pose-back)
(send *ri* :wait-interpolation)
(exit)
