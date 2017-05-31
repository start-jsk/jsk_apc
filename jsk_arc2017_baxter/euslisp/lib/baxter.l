#!/usr/bin/env roseus

;; vim: set ft=lisp:

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter.l")

(unless (find-package "JSK_ARC2017_BAXTER")
  (make-package "JSK_ARC2017_BAXTER"))


(defclass jsk_arc2017_baxter::baxter-robot
  :super jsk_2016_01_baxter_apc::baxter-robot)

(defmethod jsk_arc2017_baxter::baxter-robot
  (:spin-off-by-wrist
    (arm &key (times 10))
    (let (avs robot)
      (pushback (send self :angle-vector) avs)
      (dotimes (i times)
        (pushback (send self :rotate-wrist arm 10) avs)
        (pushback (send self :rotate-wrist arm -20) avs)
        (pushback (send self :rotate-wrist arm 20) avs)
        (pushback (send self :rotate-wrist arm -10) avs))
      avs)))
