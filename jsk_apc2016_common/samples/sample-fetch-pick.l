#!/usr/bin/env roseus

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "geometry_msgs")

(require "package://fetcheus/fetch-interface.l")

(fetch-init t)

(defun init()
      (send *fetch* :reset-pose)
      (send *fetch* :head :neck-p :joint-angle 15)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
      (send *ri* :stop-grasp)
      (send *ri* :wait-interpolation))

(defun recon()
  (setq boxes_msg
    (one-shot-subscribe
    "/cluster_indices_decomposer_target/boxes"
    jsk_recognition_msgs::BoundingBoxArray))

  (setq centroids
    (one-shot-subscribe
    "/cluster_indices_decomposer_target/centroid_pose_array"
    geometry_msgs::PoseArray))

  (setq box (car (send boxes_msg :boxes)))
  (setq center (car (send centroids :poses)))

  (setq x (* 1000 (send (send center :position) :x)))
  (setq y (* 1000 (send (send center :position) :y)))
  (setq z (* 1000 (send (send (send box :pose) :position):z)))
  t)

(defun test(x y z)
  (send *fetch* :inverse-kinematics
        (make-coords :pos (float-vector x y z)
                     :rpy (float-vector 0 (/ pi 2) 0))
        :debug-view t))

(defun test2(x y z)
  (send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y z)) :debug-view t))

(defun move (time)
  (send *ri* :angle-vector (send *fetch* :angle-vector) time)
  (send *ri* :wait-interpolation))

(defun try-grasp()
  (send *ri* :go-grasp :effort 60))

;;try grasping from above (TODO: adjust x to center of object; x+= K(dimension y) ?)
(defun trial()
  (recon)
  (test (- x 300) y (+ z 150))
  (move 6000)
  (test x y (+ z 150)) ; move end-coords x +300
  (move 3000)
  (test x y (+ z 50))
  (move 3000)
  (try-grasp)
  )

;;try grasping from front
(defun trial2()
  (recon)
  (setq z-off (/ (send (send box :dimensions) :z) 3))
  (setq z (- z z-off))
  (test2 (- x 200) y z)
  (move 6000)
  (test2 (+ x 50) y z) ; move end-coords x +250
  (move 3000)
  (try-grasp)
  ;(test2 (+ x 50) y (+ z 100)) ; move end-coords z +100
  ;(move 3000)
  )

;;TODO: avoid table!!
;;move to put x in range (700 ~ 850 ?)
;;elmer's glue: centroid too deep
