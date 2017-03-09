#!/usr/bin/env roseus

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "geometry_msgs")
(ros::load-ros-manifest "jsk_pcl_ros")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "roseus")

(require "package://fetcheus/fetch-interface.l")

(setq *co* (instance collision-object-publisher :init))
(fetch-init t)

(defun init()
      (send *fetch* :reset-pose)
      (send *fetch* :head :neck-p :joint-angle 15)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
      (send *ri* :stop-grasp)
      (send *ri* :wait-interpolation))

;;recognize position of the object.
;;x, y, z
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

  (if center
      (progn (setq x (* 1000 (send (send center :position) :x)))
	     (setq y (* 1000 (send (send center :position) :y))))
    nil)
  (if box
      (setq z (* 1000 (send (send (send box :pose) :position):z)))
    nil)
  t)

;;look at the object.
(defun look-at()
 (setq head-x (aref (send (send *fetch* :head-end-coords) :worldpos) 0)) 
 (setq head-y (aref (send (send *fetch* :head-end-coords) :worldpos) 1))
 (setq head-z (aref (send (send *fetch* :head-end-coords) :worldpos) 2))
 (send *fetch* :head :neck-y :joint-angle (rad2deg (atan (/ (- y head-y) x))))
 (send *fetch* :head :neck-p :joint-angle (rad2deg (atan (/ (- head-z z) x))))
 (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))


(defun make-table ()
  (setq cube (make-cube 800 800 800))
  (send cube :locate #f(1000 0 400))
  (send cube :set-color :green)
  (objects (list cube *fetch*))
  (send *co* :add-object  cube
	:frame-id "base_link"
	:relative-pose (make-coords :pos #f(1000 0 400) :rpy #f(0 0 0))))


;;find edge of table
(defun detect-table()
  (recon)
  (let ((distance-before) (distance-now) (i 0))
    (setq req (instance jsk_recognition_msgs::TransformScreenpointRequest :init))	  
    (send req :x 320)
    (send req :y (+ 240 (* i 10)))
    (setq res (ros::service-call "pointcloud_screenpoint/screen_to_point" req))
    (setq distance-before (* 1000 (send res :point :z)))
    (setq distance-now distance-before)
    (setq distance nil)

    (while (and (< distance-now (+ distance-before 500)) (< i 24))
      (send req :x 320)
      (send req :y (+ 240 (* i 10)))
      (setq res (ros::service-call "pointcloud_screenpoint/screen_to_point" req))
      (setq distance-before distance-now)
      (setq distance-now (* 1000 (send res :point :z)))
      (setq i (+ i 1))
      (format t "i : ~A, distance-now : ~A~%" i distance-now))
    (setq head-z (aref (send (send *fetch* :head-end-coords) :worldpos) 2))
    (if (eq i 24)
	(format t "table is too close to fetch!~%")
      ;(setq distance (sqrt (- (* distance-before distance-before) (* (- z head-z) (- z head-z))))))
      (setq distance distance-before)
    (format t "disatance between fetch and table : ~A~%" distance)))
  distance
  )

;;search for object by swinging his head.
(defun search-for-object()
  (recon)
  (if box
      (progn (recon) (look-at))
    (progn (send *fetch* :head :neck-y :joint-angle 30)
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))

  (recon)
  (if box
      (progn (recon) (look-at))
    (progn (send *fetch* :head :neck-y :joint-angle -30)
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))

  (recon)
  (if box
      (progn (recon) (look-at))
    (progn (send *fetch* :head :neck-p :joint-angle 30)
	   (send *fetch* :head :neck-y :joint-angle 0)
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))
  
  (recon)
  (if box
      (progn (recon) (look-at) t)
    (progn (format t "cannot find object!~%") nil))
  )

(defun move (time)
  (let ((test nil))
    (while (null test)
      (setq test (send *ri* :angle-vector (send *fetch* :angle-vector) time)))
    (send *ri* :wait-interpolation)))


(defun move-arm ()
  (if (detect-table)
      (progn
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector (- distance 200) y (+ z 500))) :debug-view t :rotation-axis nil :traslation-axis t)
	(move 6000)
	
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y (+ z 200))) :debug-view t)
	(move 3000)
	
	;(send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y z) :rpy (float-vector 0 (/ pi 4) 0)) :debug-view t)
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y z)) :debug-view t)
	(move 3000)))
  )
  
  
(defun try-grasp()
  (send *ri* :go-grasp :effort 60))

;;main functrion.
;;try to grasp object.
(defun trial1()
  (send *ri* :angle-vector (send *fetch* :reset-pose))
  (send *ri* :stop-grasp)
  (send *ri* :wait-interpolation)
  (format t "debug 1~%")

  (if (search-for-object)
      (progn
	(format t "debug 2~%")
	(move-arm)
	(format t "debug 3~%")
	(try-grasp))
    (format t "trial canceled.~%"))
  )

;;TODO: avoid table!!
;;move to put x in range (700 ~ 850 ?)
;;elmer's glue: centroid too deep
;;fetch mistakes his hand for white objects.
