#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "main.l")

(setq larm-pose1 #f(-3.25195 -12.4146 -12.8979 -121.707 134.143 152.82 101.294 -86.3745 -97.4707 -2.39502 94.4385 134.626 -91.3843 9.47021 0.0))
;(setq larm-pose2 #f(-3.18604 -16.4575 -0.461426 -119.246 97.8662 157.808 63.2153 -68.1152 -97.4268 -2.41699 94.043 134.648 -91.3184 11.6455 0.0))
(setq larm-pose2 #f(-3.22998 -27.0483 0.900879 -120.476 96.8994 169.409 75.4102 -66.4233 -97.4048 -2.41699 93.9771 134.648 -91.3184 12.1509 0.0))
(setq larm-pose3 #f(-3.22998 -28.916 0.483398 -122.234 77.6953 167.432 59.8975 -62.1826 -97.4268 -2.43896 93.9551 134.626 -91.3623 12.1729 0.0))
;(setq rarm-pose1 #f(-3.18604 97.4487 -2.41699 -94.5483 134.604 91.4282 9.99756 -0.021973 -14.9414 -10.2612 115.598 133.176 -154.358 61.875 72.9053))
(setq rarm-pose1 #f(-3.16406 97.4707 -2.41699 -94.5483 134.626 91.4062 10.3271 0.0 -22.6318 -8.32764 134.539 138.032 -127.244 57.1289 64.1162))
(setq rarm-pose2 #f(-3.20801 97.4707 -2.43896 -94.5044 134.561 91.4282 10.6128 0.0 -9.42627 -19.7534 118.147 120.806 -143.767 58.6011 72.356))
(setq rarm-pose3 #f(-3.20801 97.4268 -2.41699 -94.4824 134.626 91.4282 10.8765 -0.021973 28.1909 -23.9062 91.1426 58.2275 -147.151 37.9028 63.4131))
;(setq rarm-pose4 #f(-3.25195 97.4048 -2.41699 -94.4604 134.67 91.4282 9.75586 -0.021973 -14.0405 13.6011 85.8032 122.036 -125.552 52.5806 23.6206))
(setq rarm-pose4 #f(-3.12012 97.3169 -2.43896 -94.5044 134.648 91.4502 10.2832 0.0 -26.8506 7.66846 88.1323 142.581 -127.068 42.9565 28.6304))

(defun pick_ball_l ()
  (send *baxter* :angle-vector larm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)
  
  (unix::sleep 1)
  (send *ri* :start-grasp :larm)

  (send *baxter* :angle-vector larm-pose2)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  
  (send *baxter* :angle-vector larm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *ri* :fold-pose-back)
  (send *ri* :wait-interpolation)
  )

(defun pick_oreo_r ()
  (send *baxter* :angle-vector rarm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *baxter* :angle-vector rarm-pose2)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 2000)
  (send *ri* :wait-interpolation)

  (unix::sleep 2)
  (send *ri* :start-grasp :rarm)

  (send *baxter* :angle-vector rarm-pose3)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *baxter* :angle-vector rarm-pose2)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)

  (send *baxter* :angle-vector rarm-pose1)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  
  (send *baxter* :angle-vector rarm-pose4)
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

(defun pick_motion ()
  (pick_ball_l)
  (pick_oreo_r)
  (stow_ball_l)
  )
