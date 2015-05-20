;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defun move-for-verification-avs (arm)
  (let ((av-map (make-hash-table)))
    (setf (gethash :larm av-map) #f(4.74609 5.29541 -146.206 63.3472 12.8979 58.6011 -38.8916))
    (setf (gethash :rarm av-map) #f(-20.2148 -2.92236 130.056 52.4707 12.9199 71.543 32.7612))
    (send *baxter* arm :angle-vector (gethash arm av-map))
    (list (send *baxter* :angle-vector))))

(defun move-for-verification-avs-ik (arm)
  (let (goal-coords av0 av1)
    (setq goal-coords
          (case arm
                (:rarm (make-coords :pos #f(649.331 -319.89 1028.69) :rpy #f(1.30057 -0.71804 1.6585)))
                (:larm (make-coords :pos #f(690.508 171.959 1034.31) :rpy #f(-1.32998 -0.688911 -1.65887)))))
    (setq av0 (send *baxter* arm :inverse-kinematics goal-coords :rotation-axis nil)
          av1 (send *baxter* arm :inverse-kinematics goal-coords :rotation-axis t))
    (list av0 av1)))

(defun move-for-verification (arm bin)
  (let (avs)
    (pushback (avoid-shelf-pose-for-bin arm bin) avs)
    (send *irtviewer* :draw-objects)
    (setq avs (append avs (move-for-verification-avs-ik arm)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 2.2)
    (send *ri* :wait-interpolation)
    avs))
