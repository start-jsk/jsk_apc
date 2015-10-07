;; vim: set ft=lisp:

(require "package://baxtereus/baxter-util.l")

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

    ;; hard-coded pose
    (sethash :fold-pose-upper (send self :get-val 'hard-coded-pose-table) #f(0.0 -7.80029 -1.12061 -82.9248 141.438 116.477 -6.48193 8.10791 7.80029 -1.12061 82.9248 141.438 -116.477 -6.48193 -8.10791))
    (sethash :fold-pose-middle (send self :get-val 'hard-coded-pose-table) #f(0.0 48.8672 -10.437 -80.6616 148.645 80.9033 1.38428 18.8745 -48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745))
    (sethash :fold-pose-lower (send self :get-val 'hard-coded-pose-table) #f(0.0 -7.27295 52.6465 -59.9414 91.582 123.574 13.3374 159.675 7.27295 52.6465 59.9414 91.582 -123.574 13.3374 -159.675))
    (sethash :fold-pose-back (send self :get-val 'hard-coded-pose-table) #f(0.0 97.4707 -2.39502 -94.5483 134.67 91.4062 8.70117 0.0 -97.4707 -2.39502 94.5483 134.67 -91.4062 8.70117 0.0))
    (sethash :fold-to-keep-object (send self :get-val 'hard-coded-pose-table) #f(0.0 77.1899 4.37256 -86.1548 123.904 11.8213 59.8096 74.5532 -77.1899 4.37256 86.1548 123.904 -11.8213 59.8096 -74.5532))
    )
  (:fold-pose-upper (&optional (arm :arms)) (send self :hard-coded-pose :fold-pose-upper arm))
  (:fold-pose-middle (&optional (arm :arms)) (send self :hard-coded-pose :fold-pose-middle arm))
  (:fold-pose-lower (&optional (arm :arms)) (send self :hard-coded-pose :fold-pose-lower arm))
  (:fold-pose-back (&optional (arm :arms)) (send self :hard-coded-pose :fold-pose-back arm))
  (:fold-to-keep-object (&optional (arm :arms)) (send self :hard-coded-pose :fold-to-keep-object arm))
  )
