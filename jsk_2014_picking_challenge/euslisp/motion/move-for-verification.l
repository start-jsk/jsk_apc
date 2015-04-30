(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defun move-for-verification-avs (arm)
  (let ((av-l #f(22.0825 15.1392 -144.448 111.006 25.0488 2.54883 -24.895))
        (av-r #f(-22.0825 15.1392 144.448 111.006 -25.0488 2.54883 24.895)))
    (case arm
      (:larm (send *baxter* arm :angle-vector av-l))
      (:rarm (send *baxter* arm :angle-vector av-r)))
    (list (send *baxter* :angle-vector))))

(defun move-for-verification (arm)
  (let (avs)
    (setq avs (append avs (fold-pose-mid arm)))
    (send *irtviewer* :draw-objects)
    (setq avs (append avs (move-for-verification-avs arm)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.2)
    (send *ri* :wait-interpolation)
    avs))
