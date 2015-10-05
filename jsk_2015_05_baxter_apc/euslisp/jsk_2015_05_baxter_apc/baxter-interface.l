;; -*- mode: lisp;-*-

(require "package://baxtereus/baxter-interface.l")


(defun custom-baxter-init (&key ctype)
  (baxter-init :safe nil :type ctype)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send (send *baxter* :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
  (send (send *baxter* :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
  (send (send (*ri* . robot) :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
  (send (send (*ri* . robot) :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
  )
