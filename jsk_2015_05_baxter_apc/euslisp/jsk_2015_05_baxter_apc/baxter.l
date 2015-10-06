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
  (:swap-arm-av
    (av)
    (let ((_av (copy-list av)) index)
      (dotimes (i 4)
        (setq index (* i 2))
        (setf (elt _av index) (- (elt _av index))))
      _av))
  (:fold-pose-upper
    (arm)
    (let ((av-table (make-hash-table))
          (larm-av #f(-7.80029 -1.12061 -82.9248 141.438 116.477 -6.48193 8.10791)))
      (setf (gethash :larm av-table) larm-av)
      (setf (gethash :rarm av-table) (send self :swap-arm-av larm-av))
      (send self arm :angle-vector (gethash arm av-table)))
    (send self :angle-vector))
  (:fold-pose-middle
    (arm)
    (let ((av-table (make-hash-table))
          (larm-av #f(48.8672 -10.437 -80.6616 148.645 80.9033 1.38428 18.8745)))
      (setf (gethash :larm av-table) larm-av)
      (setf (gethash :rarm av-table) (send self :swap-arm-av larm-av))
      (send self arm :angle-vector (gethash arm av-table)))
    (send self :angle-vector))
  (:fold-pose-lower
    (arm)
    (let ((av-table (make-hash-table))
          (larm-av #f(-7.27295 52.6465 -59.9414 91.582 123.574 13.3374 159.675)))
      (setf (gethash :larm av-table) larm-av)
      (setf (gethash :rarm av-table) (send self :swap-arm-av larm-av))
      (send self arm :angle-vector (gethash arm av-table)))
    (send self :angle-vector))
  (:fold-pose-back
    (arm)
    (let ((av-table (make-hash-table))
          (larm-av #f(97.4707 -2.39502 -94.5483 134.67 91.4062 8.70117 0)))
      (setf (gethash :larm av-table) larm-av)
      (setf (gethash :rarm av-table) (send self :swap-arm-av larm-av))
      (send self arm :angle-vector (gethash arm av-table)))
    (send self :angle-vector))
  (:fold-to-keep-object
    (arm)
    (let ((av-table (make-hash-table))
          (larm-av #f(77.1899 4.37256 -86.1548 123.904 11.8213 59.8096 74.5532)))
      (setf (gethash :larm av-table) larm-av)
      (setf (gethash :rarm av-table) (send self :swap-arm-av larm-av))
      (send self arm :angle-vector (gethash arm av-table)))
    (send self :angle-vector))
  (:untuck-pose
    (arm)
    (let ((av-table (make-hash-table))
          (larm-av #f(5.000000 -57.000000 68.000000 111.000000 -38.000000 59.000000 -0.000000)))
      (setf (gethash arm av-table) larm-av)
      (setf (gethash arm av-table) (send self :swap-arm-av larm-av))
      (send self arm :angle-vector (gethash arm av-table)))
    (send self :angle-vector))
  )
