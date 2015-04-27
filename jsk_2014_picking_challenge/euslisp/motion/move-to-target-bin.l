(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defclass target-pick-region
  ; target pick region in the shelf
  :super propertied-object
  :slots ())
(defmethod target-pick-region
  ; local coordinates of each region
  (:init () nil)
  (:a () #f(-290 1760 570))
  (:b () #f(   0 1760 570))
  (:c () #f( 290 1760 570))
  (:d () #f(-290 1500 570))
  (:e () #f(   0 1500 570))
  (:f () #f( 290 1500 570))
  (:g () #f(-290 1250 570))
  (:h () #f(   0 1250 570))
  (:i () #f( 290 1250 570))
  (:j () #f(-290 1040 570))
  (:k () #f(   0 1040 570))
  (:l () #f( 290 1040 570))
  (:list () '(:a :b :c :d :e :f :g :h :i :j :k :l)))

(defun fold-pose-avs (arm bin)
  (let (avs)
    (setq avs (append avs (fold-pose-back (if (eq arm :larm) :rarm :larm))))  ; fold oposite arm
    (setq avs (append avs (fold-pose-mid arm)))
    (cond ((find bin '(:a :b :c)) (setq avs (append avs (fold-pose-up arm))))
          ((find bin '(:d :e :f :g :i)) (setq avs (append avs (fold-pose-mid arm))))
          ((find bin '(:h :j :k :l)) (setq avs (append avs (fold-pose-low arm))))
          (t nil))
    avs))

(defun move-to-target-bin-avs (arm bin)
  (let (c (region (instance target-pick-region :init)))
    (setq c (send *pod* :transform-vector (send region bin)))
    (send *baxter* arm :inverse-kinematics (make-cascoords :pos c) :rotation-axis t)
    (list (send *baxter* :angle-vector))))

(defun move-to-target-bin (arm bin &optional (ctype :default-controller))
  (let (avs)
    (setq avs (append avs (fold-pose-avs arm bin)))
    (setq avs (append avs (move-to-target-bin-avs arm bin)))
    (send *ri* :angle-vector-sequence avs :fast ctype 0 :scale 2.2)
    (send *ri* :wait-interpolation ctype)
    avs))
