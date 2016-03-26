(load "main.l")

(setq *oreointote* (make-coords :pos #f(785 -35 -293)))

(defun reset2oreo ()
  (send *baxter* :rarm :inverse-kinematics *oreointote* :rotation-axis nil :debug-view t))

(setq *viapoint* (make-coords :pos #f(850 0 1300) :rpy (float-vector (/ pi 4.0) 0 0)))

(defun oreo2via ()
  (send *baxter* :rarm :inverse-kinematics *viapoint* :rotation-axis t :debug-view t))

(setq *goal* (make-coords :pos #f(1200 0 1300)))

(defun via2bin ()
  (send *baxter* :rarm :inverse-kinematics *goal* :rotation-axis t :debug-view t))

(defun back_bin2via ()
  (send *baxter* :rarm :inverse-kinematics *viapoint* :rotation-axis t :debug-view t))

(defun back_via2reset ()
  (send *baxter* :reset-pose))
