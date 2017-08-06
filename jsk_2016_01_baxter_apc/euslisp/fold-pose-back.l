#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter-interface.l")

(ros::roseus "fold_pose_back")

(setq l/r (car (last lisp::*eustop-argument*)))
(setq arm
      (cond
        ((string= l/r "left") '(:larm))
        ((string= l/r "right") '(:rarm))
        (t '(:larm :rarm))
        ))

(jsk_2016_01_baxter_apc::baxter-init)

(send *ri* :angle-vector
      (send *baxter* :fold-pose-back arm)
      10000)
(dolist (_arm arm)
  (send *ri* :gripper-servo-on _arm)
  )
(send *ri* :wait-interpolation)
(exit)
