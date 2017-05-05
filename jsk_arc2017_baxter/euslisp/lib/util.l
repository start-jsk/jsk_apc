#!/usr/bin/env roseus

;; TODO: move to jsk_arc2017_baxter::baxter-robot
(defun get-opposite-arm (arm) (if (eq arm :larm) :rarm :larm))

;; TODO: move to jsk_arc2017_baxter::baxter-interface
(defun get-arm-controller (arm)
  (cond ((eq arm :rarm) :rarm-controller)
        ((eq arm :larm) :larm-controller)
        (t nil)))
