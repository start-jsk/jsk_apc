#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-init.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-motions.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-test.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-visualization.l")

(setq *rarm-bin-lists* (list :c :f :i :l))
(setq *larm-bin-lists* (list :a :b :d :e :g :h :j :k))

(defun take-picture(arm bin )
  (move-to-target-bin arm bin)
  (unix::sleep 5)
  (speak-en (format nil "Take a picture of ~A" (key-to-str bin)))
  (warn (format nil "Take a picture of ~A in ~A" (key-to-str bin) (key-to-str arm)))
  (ros::service-call (format nil "/~A_image_saver/save" (key-to-str arm)) (instance std_srvs::EmptyRequest :init))
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
