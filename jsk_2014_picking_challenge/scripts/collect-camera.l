#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-test.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-visualization.l")

(setq *rarm-bin-lists* (list :c :f :i :l))
(setq *larm-bin-lists* (list :a :b :d :e :g :h :j :k))

(defun take-picture(arm bin )
  (move-to-target-bin arm bin)
  (unix::sleep 5)
  (speak-en (format nil "Take a picture of ~A" (symbol-to-str bin)))
  (warn (format nil "Take a picture of ~A in ~A" (symbol-to-str bin) (symbol-to-str arm)))
  (ros::service-call (format nil "/~A_image_saver/save" (symbol-to-str arm)) (instance std_srvs::EmptyRequest :init))
  (unix::sleep 5)
  )

(defun main()
  (dolist (bin *rarm-bin-lists*)
    (take-picture :rarm bin)
    )
  (dolist (bin *larm-bin-lists*)
    (take-picture :larm bin)
    )
  )


(warn "~% Commands ~%")
(warn "(apc-init)")
(warn "(main)")
