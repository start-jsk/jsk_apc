#!/usr/bin/env roseus

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "geometry_msgs")
(ros::load-ros-manifest "jsk_pcl_ros")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "roseus")

(require "package://fetcheus/fetch-interface.l")

(ros::advertise "stop_moveit" std_msgs::string 100)
(setq msg (instance std_msgs::string :init))
(send msg :data "stop")



(setq *co* (instance collision-object-publisher :init))
(fetch-init t)


;; initialize pose of the real fetch robot
(defun init()
      (send *fetch* :reset-pose)
      (send *fetch* :head :neck-p :joint-angle 15) ;; see down
      (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
      (send *ri* :stop-grasp)
      (send *ri* :wait-interpolation))


;; recognize position of the target object.
;; frame_id : /base_link
(defun recog-target()
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
    ;;(format t "cannot find target!~%")))
    ))


;; look at the target object if fetch has already found it.
(defun look-at()
 (setq head-x (aref (send (send *fetch* :head-end-coords) :worldpos) 0)) 
 (setq head-y (aref (send (send *fetch* :head-end-coords) :worldpos) 1))
 (setq head-z (aref (send (send *fetch* :head-end-coords) :worldpos) 2))
 (send *fetch* :head :neck-y :joint-angle (rad2deg (atan (/ (- y head-y) x))))
 (send *fetch* :head :neck-p :joint-angle (rad2deg (atan (/ (- head-z z) x))))
 (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)
 (send *ri* :wait-interpolation :head-controller))


;; search for a target object by swinging his head.
(defun search-for-object()
  (recog-target)
  (if box
      (progn (look-at) (format t "111~%"))
    (progn (send *fetch* :head :neck-y :joint-angle 30) ;; see left
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))

  (recog-target)
  (if box
      (progn (look-at) (format t "222~%"))
    (progn (send *fetch* :head :neck-y :joint-angle -30) ;; see right 
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))

  (recog-target)
  (if box
      (progn (look-at) (format t "333~%"))
    (progn (send *fetch* :head :neck-p :joint-angle -30)
	   (send *fetch* :head :neck-y :joint-angle 0) ;; see up
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))
  
  (recog-target)
  (if box
      (progn (look-at) (format t "444~%"))
    (progn (send *fetch* :head :neck-p :joint-angle 30) ;; see down
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))

  (recog-target)
  (if box
      (progn (look-at) (format t "find object!~%") t)
    (progn (format t "cannot find object!~%")
	   (send *fetch* :head :neck-p :joint-angle 0) ;; see front
	   (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)))
  )


;; order the real fetch robot to follow the joint angle of the model robot
(defun move (time &optional (trial-number 100))
  (let ((test nil) (i 0))
    (while (and (null test) (< i trial-number))
      (setq i (+ i 1))
      (setq test (send *ri* :angle-vector (send *fetch* :angle-vector) time)))
    (send *ri* :wait-interpolation)))


(defun try-grasp()
  (send *ri* :go-grasp :effort 60))


;;main function.
;;try to pick object.
(defun pick()
  (send *ri* :angle-vector (send *fetch* :reset-pose))
  (send *ri* :stop-grasp)
  (send *ri* :wait-interpolation)

  (if (search-for-object)
      (progn
	(ros::publish "stop_moveit" msg)
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector (- x 200) y (+ z 200))) :debug-view t)
	(move 10000)
	(let ((flag nil) (i 0))
	  (while (null flag)
	    (send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y (+ z i))))
	    (setq flag (move 5000 1))
	    (setq i (+ i 5))))
	(send *ri* :wait-interpolation)
	
	(try-grasp)
	
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector (- x 200) y (+ z 200))))
	(move 1000)
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector (- x 300) y (+ z 100))))
	(move 2000)
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector 350 600 (+ z 100))) :debug-view t)
	(move 5000)
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector 350 600 100)) :debug-view t :rotation-axis nil)
	(move 5000)
	(send *ri* :stop-grasp)
	(init))
    (format t "trial canceled.~%"))
  )

;;elmer's glue: centroid too deep
;;fetch mistakes his hand for collision object.
