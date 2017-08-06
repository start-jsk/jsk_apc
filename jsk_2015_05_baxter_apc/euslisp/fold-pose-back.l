#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/baxter-interface.l")

(ros::roseus "fold_pose_back")

(setq l/r (car (last lisp::*eustop-argument*)))
(setq arm
      (cond
        ((string= l/r "left") :larm)
        ((string= l/r "right") :rarm)
        (t :arms)
        ))

(jsk_2015_05_baxter_apc::baxter-init)
(send *ri* :fold-pose-back arm)
(send *ri* :wait-interpolation)
(exit)
