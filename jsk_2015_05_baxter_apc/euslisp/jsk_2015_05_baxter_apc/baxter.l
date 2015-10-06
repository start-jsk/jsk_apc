;; vim: set ft=lisp:

(require "package://baxtereus/baxter.l")

(unless (find-package "JSK_2015_05_BAXTER_APC")
  (make-package "JSK_2015_05_BAXTER_APC"))

(defclass jsk_2015_05_baxter_apc::baxter-robot
  :super baxter-robot
  :slots ())
(defmethod jsk_2015_05_baxter_apc::baxter-robot
  (:init
    (&rest args)
    (send-super* :init args)
    (send (send self :rarm :end-coords)
          :newcoords (make-coords :pos #f(50 0 310)
                                  :rpy (float-vector 0 -pi/2 0)))
    (send (send self :larm :end-coords)
          :newcoords (make-coords :pos #f(50 0 310)
                                  :rpy (float-vector 0 -pi/2 0)))
    )
  )
