(defun detect-bin-position ()
  (let
      ((coords-left (make-coords :pos (v+ #f(790 0 -185) #f(300 300 200))))
       (coords-right (make-coords :pos (v+ #f(790 0 -185) #f(-300 -300 200)))))
    (send *baxter* :rarm :inverse-kinematics coords-right :rotation-axis nil)
    (send *baxter* :larm :inverse-kinematics coords-left :rotation-axis nil)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 5000 nil 0)
    (send *ri* :wait-interpolation)))

