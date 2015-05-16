;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defun move-for-verification-avs (arm)
  (let ((av-l #f(4.74609 5.29541 -146.206 63.3472 12.8979 58.6011 -38.8916))
        (av-r #f(-4.74609 5.29541 146.206 63.3472 -12.8979 58.6011 38.8916)))
    (case arm
      (:larm (send *baxter* arm :angle-vector av-l))
      (:rarm (send *baxter* arm :angle-vector av-r)))
    (list (send *baxter* :angle-vector))))

(defun move-for-verification (arm bin)
  (let (avs)
    (pushback (avoid-shelf-pose-for-bin arm bin) avs)
    (send *irtviewer* :draw-objects)
    (setq avs (append avs (move-for-verification-avs arm)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 2.2)
    (send *ri* :wait-interpolation)
    avs))
