#!/usr/bin/env roseus

(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(ros::roseus "detect_bin_position")

(defun detect-bin-position ()
  (let
      ((coords-left (make-coords :pos (v+ #f(790 0 -185) #f(300 300 200))))
       (coords-right (make-coords :pos (v+ #f(790 0 -185) #f(-300 -300 200)))))
    (send *baxter* :rarm :inverse-kinematics coords-right :rotation-axis nil)
    (send *baxter* :larm :inverse-kinematics coords-left :rotation-axis nil)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 5000 nil 0)
    (send *ri* :wait-interpolation)))

(warn "~% Commands ~%")
(warn "First, you need to initialize *ri* and *baxter*, if you haven't done yet.~%")
(warn "(jsk_arc2017_baxter::arc-init) : initialize *ri* and *baxter*~%")
(warn "(detect-bin-position) : detect bin position~%")
