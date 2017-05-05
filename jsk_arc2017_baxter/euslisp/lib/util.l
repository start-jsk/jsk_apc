#!/usr/bin/env roseus

(defun get-opposite-arm (arm) (if (eq arm :larm) :rarm :larm))

(defun get-arm-controller (arm)
  (cond ((eq arm :rarm) :rarm-controller)
        ((eq arm :larm) :larm-controller)
        (t nil)))
