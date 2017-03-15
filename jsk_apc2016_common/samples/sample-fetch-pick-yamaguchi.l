#!/usr/bin/env roseus

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "geometry_msgs")
(ros::load-ros-manifest "jsk_pcl_ros")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "roseus")

(require "package://fetcheus/fetch-interface.l")

(setq *co* (instance collision-object-publisher :init))
(fetch-init t)


;; initialize pose of the real fetch robot
(defun init()
      (send *fetch* :reset-pose)
      (send *fetch* :head :neck-p :joint-angle 15) ;; see down
      (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
      (send *ri* :stop-grasp)
      (send *ri* :wait-interpolation))


;; recognize position of the target object.
;; frame_id : /base_link
(defun recon-target()
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
    (format t "cannot find target!~%")))


;; recognize the table through hsi color filter
;; frame_id : /head_camera_depth_optical_frame
;; TODO : coordinate transformation (/head_camera_depth_optical_frame -> /base_link)
(defun recon-table()
  (setq table-boxes-msg
	(one-shot-subscribe
	 "/head_camera/depth_registered/boxes"
	 jsk_recognition_msgs::BoundingBoxArray))

  (setq table-centroids
  	(one-shot-subscribe
  	 "/head_camera/depth_registered/cluster_decomposer/centroid_pose_array"
  	 geometry_msgs::PoseArray))
  (setq table-box (car (send table-boxes-msg :boxes)))
  (setq table-center (car (send table-centroids :poses)))

  (if table-center
      (progn (setq table-x (* 1000 (send (send table-center :position) :x)))
	     (setq table-y (* 1000 (send (send table-center :position) :y))))
    nil)
  (if table-box
      (setq table-z (* 1000 (send (send (send table-box :pose) :position):z)))
    nil)
  (if (< table-z 1000)
      (setq distance table-z)
    (format t "cannot recognize table or table is too far from fetch!~%")))


;; look at the target object if fetch has already found it.
(defun look-at()
 (setq head-x (aref (send (send *fetch* :head-end-coords) :worldpos) 0)) 
 (setq head-y (aref (send (send *fetch* :head-end-coords) :worldpos) 1))
 (setq head-z (aref (send (send *fetch* :head-end-coords) :worldpos) 2))
 (send *fetch* :head :neck-y :joint-angle (rad2deg (atan (/ (- y head-y) x))))
 (send *fetch* :head :neck-p :joint-angle (rad2deg (atan (/ (- head-z z) x))))
 (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller))


;; make a virtual table in front of fetch.
;; the table is added to collision-object in this function.
(defun make-table (pos-x pos-y pos-z)
  (let (plate leg1 leg2 leg3 leg4)
    (setq plate (make-cube 600 800 20))
    (send plate :set-color :brown)
    (send plate :locate #f(0 0 790))
    (setq leg1 (make-cylinder 10 780))
    (send leg1 :locate #f(280 380 0))
    (setq leg2 (make-cylinder 10 780))
    (send leg2 :locate #f(280 -380 0))
    (setq leg3 (make-cylinder 10 780))
    (send leg3 :locate #f(-280 380 0))
    (setq leg4 (make-cylinder 10 780))
    (send leg4 :locate #f(-280 -380 0))
    (send plate :assoc leg1)
    (send plate :assoc leg2)
    (send plate :assoc leg3)
    (send plate :assoc leg4)
    (if (not (boundp 'table))
	(setq table (instance bodyset :init (make-cascoords)
			              :bodies (list plate leg1 leg2 leg3 leg4))))
    (send table :move-to (make-coords :pos (float-vector pos-x pos-y pos-z)) :world) ;; locate table in front of fetch
    (objects (list *fetch* table))
    (send *co* :add-object  table
	  :frame-id "base_link"
	  :relative-pose (make-coords :pos (float-vector pos-x pos-y pos-z) :rpy #f(0 0 0)))))


;; delete the virtual table made in (make-table)
(defun delete-table ()
  (if (boundp 'table)
      (send *co* :delete-object table))
  (objects (list *fetch*)))


;; find edge of the table
(defun recon-table-edge()
  (recon-target)
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


;; search for a target object by swinging his head.
(defun search-for-object()
  (recon-target)
  (if box
      (progn (recon-target) (look-at))
    (progn (send *fetch* :head :neck-y :joint-angle 30) ;; see left
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))

  (recon-target)
  (if box
      (progn (recon-target) (look-at))
    (progn (send *fetch* :head :neck-y :joint-angle -30) ;; see right 
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))

  (recon-target)
  (if box
      (progn (recon-target) (look-at))
    (progn (send *fetch* :head :neck-p :joint-angle -30)
	   (send *fetch* :head :neck-y :joint-angle 0) ;; see up
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))
  
  (recon-target)
  (if box
      (progn (recon-target) (look-at))
    (progn (send *fetch* :head :neck-p :joint-angle 30) ;; see down
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))

  (recon-target)
  (if box
      (progn (recon-target) (look-at) t)
    (progn (format t "cannot find object!~%")
	   (send *fetch* :head :neck-p :joint-angle 0) ;; see front
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))
  )


;; order the real fetch robot to follow the joint angle of the model robot
(defun move (time)
  (let ((test nil))
    (while (null test)
      (setq test (send *ri* :angle-vector (send *fetch* :angle-vector) time)))
    (send *ri* :wait-interpolation)))


;; move arm to the target object if fetch recognize table
(defun move-arm-with-certain-path ()
  (send *fetch* :inverse-kinematics (make-coords :pos (float-vector (- distance 200) y (+ z 500))) :debug-view t :rotation-axis nil :traslation-axis t) ;; move hand to over the object
  (move 6000)
  
  (send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y (+ z 200))) :debug-view t) ;; move hand to over the object (nearer than the previous line)
  (move 3000)
  
  ;(send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y z) :rpy (float-vector 0 (/ pi 4) 0)) :debug-view t)
  (send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y z)) :debug-view t) ;; move hand to the very place where the target object exists
  (move 3000))
  
  
(defun try-grasp()
  (send *ri* :go-grasp :effort 60))


;;main function.
;;try to grasp object.
(defun trial1()
  (send *ri* :angle-vector (send *fetch* :reset-pose))
  (send *ri* :stop-grasp)
  (send *ri* :wait-interpolation)
  (format t "debug 1~%")

  (if (search-for-object)
      (progn
	(format t "debug 2~%")
	(if recon-table-edge
	    (move-with-certain-path))
	(format t "debug 3~%")
	(try-grasp))
    (format t "trial canceled.~%"))
  )

(defun trial2()
  (send *ri* :angle-vector (send *fetch* :reset-pose))
  (send *ri* :stop-grasp)
  (send *ri* :wait-interpolation)
  (format t "debug 1~%")

  (if (search-for-object)
      (progn
	(format t "debug 2~%")
	(if recon-table
	    (progn
	      (make-table x y z) ;;use recon-table and tf !! (x -> table-x)
	      (send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y z)) :debug-view t)
	      (move 10000)))
	(format t "debug 3~%")
	(try-grasp))
    (format t "trial canceled.~%"))
  )



;;TODO: avoid table!!
;;move to put x in range (700 ~ 850 ?)
;;elmer's glue: centroid too deep
;;fetch mistakes his hand for white objects.
