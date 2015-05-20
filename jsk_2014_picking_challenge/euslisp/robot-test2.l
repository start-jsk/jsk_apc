#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_test")

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-communication.l")

(defun test-try-to-pick-object (arm bin pattern depth)
  (move-end-pos-with-interpolation arm :x depth :time 1000)

  ;; move left or right
  (case pattern
    (:right-most
     (move-end-pos-with-interpolation arm :y -80 :time 1000)
     )
    (:left-most
     (move-end-pos-with-interpolation arm :y 80 :time 1000)
     )
    (:right
     (move-end-pos-with-interpolation arm :y -50 :time 1000)
     )
    (:left
     (move-end-pos-with-interpolation arm :y 50 :time 1000)
     )
    (:middle
     ;; none
     )
    (:middle-left
     (move-end-pos-with-interpolation arm :y 20 :time 1000)
     )
    (:middle-right
     (move-end-pos-with-interpolation arm :y -20 :time 1000)
     )
    (:rotate-and-right
     (move-end-pos-with-interpolation arm :z -50)
     (rotate-wrist arm -90)
     (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
     (send *ri* :wait-interpolation)
     (move-end-pos-with-interpolation arm :y -80 :time 1000))
    (:rotate-and-left
     (move-end-pos-with-interpolation arm :z -50)
     (rotate-wrist arm 90)
     (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
     (send *ri* :wait-interpolation)
     (move-end-pos-with-interpolation arm :y 80 :time 1000))
    )

  (case pattern
    ((:right-most :left-most :right :left :middle :middle-left :middle-right)
     (move-end-pos-with-interpolation arm :z -120)
     )
    ((:rotate-and-left :rotate-and-right)
     ;; none
     )
    )

  (send *irtviewer* :draw-objects)
  (unix::sleep 1)

  (send *ri* :angle-vector (send *baxter* arm :move-end-pos #f(0 0 30)) 1500)
  (send *ri* :wait-interpolation)
  (insert-to-pick-object-avs arm bin)
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  )

(defun test-pick-object-depth (arm bin)
  ;; insert arm to target bin
  (dolist (av (insert-to-pick-object-avs arm bin))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector av 3000)
    (send *ri* :wait-interpolation))
  (send *ri* :start-grasp arm)
  (dolist (pattern '((:middle . 0) (:middle 80) (:middle 150) (:middle 200)))
    (format t "pattern ~A, depth ~A" (car pattern) (cdr pattern))
    (test-try-to-pick-object arm bin pattern depth))
  (send *ri* :stop-grasp arm)
  )

(defun test ()
  (send *ri* :stop-grasp)
  (dolist (bins '((:a . :c) (:d . :f) (:g . :i) (:j . :l)))
    (format t "move to ~A and ~A~%" (car bins) (cdr bins))
    (when (member (car bins) '(:a :d))
      (fold-to-keep-object-av :larm)
      (fold-to-keep-object-av :rarm)
      (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
      (send *ri* :wait-interpolation))
    (move-to-target-bin :larm (car bins))
    (move-to-target-bin :rarm (cdr bins))
    (test-pick-object-depth :larm (car bins))
    (test-pick-object-depth :rarm (car bins))
    ))

(apc-init)
(test)
