#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "main.l")

(setq larm-pose1 #f(-3.25195 -12.4146 -12.8979 -121.707 134.143 152.82 101.294 -86.3745 -97.4707 -2.39502 94.4385 134.626 -91.3843 9.47021 0.0))
(setq larm-pose2 #f(-3.18604 -16.4575 -0.461426 -119.246 97.8662 157.808 63.2153 -68.1152 -97.4268 -2.41699 94.043 134.648 -91.3184 11.6455 0.0))
(setq larm-pose3 #f(-3.22998 -28.916 0.483398 -122.234 77.6953 167.432 59.8975 -62.1826 -97.4268 -2.43896 93.9551 134.626 -91.3623 12.1729 0.0))
(setq rarm-pose1 #f(-3.18604 97.4487 -2.41699 -94.5483 134.604 91.4282 9.99756 -0.021973 -14.9414 -10.2612 115.598 133.176 -154.358 61.875 72.9053))
(setq rarm-pose2 #f(-3.20801 97.4048 -2.39502 -94.4604 134.626 91.4062 10.5029 0.0 30.7397 -14.6338 96.7017 58.1836 -158.818 40.979 66.0718))
(setq rarm-pose3 #f(-3.25195 97.4048 -2.41699 -94.4604 134.67 91.4282 9.75586 -0.021973 -14.0405 13.6011 85.8032 122.036 -125.552 52.5806 23.6206))

(defun pick_ball_l ()
  (send *baxter* :angle-vector larm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *ri* :start-grasp :larm)

  (send *baxter* :angle-vector larm-pose2)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)
  
  (send *baxter* :angle-vector larm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *ri* :fold-pose-back)
  (send *ri* :wait-interpolation)
  )

(defun pick_ball_r ()
  (send *baxter* :angle-vector rarm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *ri* :start-grasp :rarm)

  (send *baxter* :angle-vector rarm-pose2)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *baxter* :angle-vector rarm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *baxter* :angle-vector rarm-pose3)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *ri* :stop-grasp :rarm)
  (unix::sleep 5)

  (send *ri* :fold-pose-back)
  (send *ri* :wait-interpolation)
  )

(defun stow_ball_l ()
  (send *baxter* :angle-vector larm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *baxter* :angle-vector larm-pose3)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *ri* :stop-grasp :larm)
  (unix::sleep 5)

  (send *baxter* :angle-vector larm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *ri* :fold-pose-back)
  (send *ri* :wait-interpolation)
  )
