(defun assert-arm (arm)
  (if (and
       (not (eq arm :larm))
       (not (eq arm :rarm))
       (format t "[assertion] arm is not :larm or :rarm")
       ))
  )

(defun draw-objects ()
  (send *irtviewer* :draw-objects))
(defun da () (draw-objects))

(defun ri ()
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  )

(defun draw-coord (arm)
  (assert-arm arm)
  (send *baxter* :larm :end-coords :draw-on :flush t)
  )
(defun dc (arm)
  (draw-coord arm))


(defun rot (arm &key (x 0) (y 0) (z 0))
  (assert-arm arm)
  (send *baxter* arm :move-end-rot x :x)
  (send *baxter* arm :move-end-rot y :y)
  (send *baxter* arm :move-end-rot z :z)
  (dc arm)
  (da)
  )

(defun mttb (arm bin &key (fast t))
  (move-to-target-bin arm bin :fast fast)
  )

(defun pick (arm bin &key (to-see-if-grabbed nil))
  (pick-object arm bin :to-see-if-grabbed to-see-if-grabbed)
  )
