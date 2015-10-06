;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/motion/robot-motion-common.l")

(defun move-to-target-bin-av (arm bin)
  (cond ((find bin '(:a :b :c)) (send *baxter* :fold-pose-upper arm))
        ((find bin '(:d :e :f :g :i)) (send *baxter* :fold-pose-middle arm))
        ((find bin '(:j :k :l :h)) (send *baxter* :fold-pose-lower arm))
        (t (return-from move-to-target-bin-av nil)))
  ;; move arm to the position 90mm away from the target bin
  (send *baxter* arm :inverse-kinematics (make-cascoords :pos (v+ (send *pod* bin) #f(-90 0 0))) :rotation-axis t)
  (send *baxter* :angle-vector))

(defun move-to-target-bin (arm bin &key (fast t))
  (let (avs)
    (pushback (fold-to-keep-object-av arm) avs)
    (pushback (avoid-shelf-pose-for-bin arm bin) avs)
    (pushback (move-to-target-bin-av arm bin) avs)
    (send *irtviewer* :draw-objects)

    (if fast
      (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 2.8)
      (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
      )
    (send *ri* :wait-interpolation)
    (force-to-reach-goal :arm arm :stop 3)
    avs))
