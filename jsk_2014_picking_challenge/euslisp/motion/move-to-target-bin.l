;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defun move-to-target-bin-av (arm bin)
  (cond ((find bin '(:a :b :c)) (fold-pose-up arm))
        ((find bin '(:d :e :f :g :i)) (fold-pose-mid arm))
        ((find bin '(:j :k :l :h)) (fold-pose-low arm))
        (t (return-from move-to-target-bin-av nil)))
  ;; move arm to the position 90mm away from the target bin
  (send *baxter* arm :inverse-kinematics (make-cascoords :pos (v+ (send *pod* bin) #f(-90 0 0))) :rotation-axis t)
  (send *baxter* :angle-vector))

(defun move-to-target-bin (arm bin)
  (let (avs)
    (send *baxter* arm :move-end-pos #f(-100 0 0) :world :revert-if-fail nil :rotation-axis nil)
    (pushback (fold-to-keep-object-av arm) avs)
    (pushback (avoid-shelf-pose-for-bin arm bin) avs)
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 2.8)
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector (move-to-target-bin-av arm bin) 1000)
    (send *irtviewer* :draw-objects)
    (send *ri* :wait-interpolation))
