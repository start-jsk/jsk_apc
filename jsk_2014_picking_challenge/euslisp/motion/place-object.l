(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defun place-object-avs (arm)
  (let (binc avs)
    (setq binc (make-cascoords :pos (send *orderbin* :transform-vector #f(0 0 300))
                               :rot #2f((-3.6e-06 0 -1) (0 1 0) (1 0 -3.6e-06))))
    (pushback (send *baxter* arm :inverse-kinematics binc :rotation-axis :z :revert-if-fail nil) avs)
    (pushback (send *baxter* arm :move-end-pos #f(0 0 -250) :world :revert-if-fail nil) avs)
    avs))

(defun place-object (arm &optional (ctype :default-controller))
  (let (avs)
    (setq avs (append avs (fold-pose-back (if (eq arm :larm) :rarm :larm))))  ; fold oposite arm
    (setq avs (append avs (fold-pose-up arm)))
    (setq avs (append avs (untuck-pose arm)))
    (setq avs (append avs (place-object-avs arm)))
    (send *ri* :angle-vector-sequence avs :fast ctype 0 :scale 2.2)
    (send *ri* :wait-interpolation ctype)
    avs))
