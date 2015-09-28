#!/usr/bin/env roseus
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/gripper-control.l")

(defun init ()
  (baxter-init)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *baxter* :locate #f(0 0 950) :world)
  (objects (list *baxter*)))

(defun start ()
  (send *ri* :go-grasp))

(defun stop ()
  (send *ri* :stop-grasp))

(defun s ()
  (send *ri* :go-grasp))

(defun r ()
  (send *ri* :stop-grasp))

(defun go (angles)
  (send *ri* :angle-vector (send *baxter* :angle-vector angles)))

(init)
