(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defclass target-pick-region
  ; target pick region in the shelf
  :super propertied-object
  :slots ())
(defmethod target-pick-region
  ; local coordinates of each region
  (:init () nil)
  (:a () #f(-280 1660 490))
  (:b () #f(   0 1660 490))
  (:c () #f( 280 1660 490))
  (:d () #f(-280 1430 490))
  (:e () #f(   0 1430 490))
  (:f () #f( 280 1430 490))
  (:g () #f(-280 1200 490))
  (:h () #f(   0 1200 490))
  (:i () #f( 280 1200 490))
  (:j () #f(-280  940 490))
  (:k () #f(   0  940 490))
  (:l () #f( 280  940 490))
  (:list () '(:a :b :c :d :e :f :g :h :i :j :k :l)))

(defun fold-pose-avs (arm bin)
  (let (avs)
    (setq avs (append avs (fold-pose-mid arm)))
    (cond ((find bin '(:a :b :c)) (setq avs (append avs (fold-pose-up arm))))
          ((find bin '(:d :e :f :g :i)) (setq avs (append avs (fold-pose-mid arm))))
          ((find bin '(:j :k :l :h)) (setq avs (append avs (fold-pose-low arm))))
          (t nil))
    avs))

(defun move-to-target-bin-avs (arm bin)
  (let (c (region (instance target-pick-region :init)))
    (setq c (send *pod* :transform-vector (send region bin)))
    (send *baxter* arm :inverse-kinematics (make-cascoords :pos c) :rotation-axis t)
    (list (send *baxter* :angle-vector))))

(defun move-to-target-bin (arm bin)
  (let (avs)
    (pushback (look-at-other-side arm) avs)
    (setq avs (append avs (fold-pose-back (case arm (:larm :rarm) (:rarm :larm)))))  ; fold oposite arm
    (fold-pose-avs arm bin)
    (setq avs (append avs (fold-pose-mid arm)))
    (setq avs (append avs (move-to-target-bin-avs arm bin)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.2)
    (send *ri* :wait-interpolation)
    avs))
