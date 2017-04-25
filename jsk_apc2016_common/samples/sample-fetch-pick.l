#!/usr/bin/env roseus

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "geometry_msgs")
(ros::load-ros-manifest "sensor_msgs")
;(ros::load-ros-manifest "jsk_pcl_ros")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "roseus")
(ros::load-ros-package "force_proximity_ros")

(require "package://fetcheus/fetch-interface.l")

(ros::advertise "stop_moveit" std_msgs::string 100)
(setq msg (instance std_msgs::string :init))

(setq *co* (instance collision-object-publisher :init))
(fetch-init t)

(load "package://jsk_apc2016_common/samples/publish_FA_I_sensor_marker.l")

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
  (let ((flag t))
    (if (and center box)
	(progn (look-at) (setq flag nil) (format t "111~%"))
      (progn (send *fetch* :head :neck-y :joint-angle 30) ;; see left
	     (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)
	     (recog-target)))

    (if flag
	(if (and center box)
	    (progn (look-at) (setq flag nil) (format t "222~%"))
	  (progn (send *fetch* :head :neck-y :joint-angle -30) ;; see right
		 (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)
		 (recog-target))))

    (if flag
	(if (and center box)
	    (progn (look-at) (setq flag nil) (format t "333~%"))
	  (progn (send *fetch* :head :neck-p :joint-angle -30)
		 (send *fetch* :head :neck-y :joint-angle 0) ;; see up
		 (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)
		 (recog-target))))

    (if flag
	(if (and center box)
	    (progn (look-at) (setq flag nil) (format t "444~%"))
	  (progn (send *fetch* :head :neck-p :joint-angle 30) ;; see down
		 (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller)
		 (recog-target))))

    (if flag
	(if (and center box)
	    (progn (look-at) (setq flag nil))
	  (progn (send *fetch* :head :neck-p :joint-angle 0) ;; see front
		 (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000 :head-controller))))

    (if (null flag)
	(progn (format t "find object!~%") t)
      (progn (format t "cannot find object!~%") nil))))


;; order the real fetch robot to follow the joint angle of the model robot
(defun move (time &optional (trial-number 100))
  (let ((test nil) (i 0))
    (while (and (null test) (< i trial-number))
      (setq i (+ i 1))
      (setq test (send *ri* :angle-vector (send *fetch* :angle-vector) time)))
    (send *ri* :wait-interpolation)))


(defun try-grasp()
  (send *ri* :go-grasp :effort 60))


(defun grope-from-front()
  ;before use this function, do below
  ;(send *ri* :angle-vector (send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector (- x 100) y  table-height) (float-vector 0 width-sum i)))))

  ;;proximity1,2 are for width.
  ;;proximity0,3,4 are for depth.
  (echo-proximity)
  (unix:sleep 1)

  (let (
	;(proximity0-init-value) (proximity1-init-value) (proximity2-init-value)
	(proximity0-raw) (proximity1-raw) (proximity2-raw)
	(proximity0 0) (proximity1 0) (proximity2 0)
	(i 0) (width-sum 0))

    (FA-I-calibration)
    ;(ros::spin-once)
    ;;sensor calibration (far away from the target)
    ;; (setq proximity0-init-value (send proximity0-obj :average))
    ;; (setq proximity1-init-value (send proximity1-obj :average))
    ;; (setq proximity2-init-value (send proximity2-obj :average))

    (setq grasp-flag-front nil)
    ;(while (null grasp-flag-front)
    (while (< proximity0 15000)
      (setq proximity0-raw (send proximity0-obj :average))
      (setq proximity1-raw (send proximity1-obj :average))
      (setq proximity2-raw (send proximity2-obj :average))

      (setq proximity0 (- proximity0-raw proximity0-init-value))
      (setq proximity1 (- proximity1-raw proximity1-init-value))
      (setq proximity2 (- proximity2-raw proximity2-init-value))

      (format t "i : ~A~%" i)
      (format t "proximity0 : ~A~%" proximity0)
      (format t "proximity1 : ~A~%" proximity1)
      (format t "proximity2 : ~A~%" proximity2)

      (setq width-sum (+ width-sum (/ (- proximity2 proximity1) 100)))
      (setq i (+ i 10))

      (send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector (- x 100) y table-height) (float-vector i width-sum 0))))
      (move 1000)

      (ros::spin-once)))

  (try-grasp))


(defun FA-I-calibration()
  (ros::spin-once)
  (setq proximity0-init-value (send proximity0-obj :average))
  (setq proximity1-init-value (send proximity1-obj :average))
  (setq proximity2-init-value (send proximity2-obj :average))
  (setq proximity3-init-value (send proximity3-obj :average))
  (setq proximity4-init-value (send proximity4-obj :average)))


(defun calc-arrow-size-in-rviz (proximity-msg init-value)
  (- 10 (* 10 (expt (max (- (send proximity-msg :average) init-value) 1) (/ -1.0 5))))) 

;; TODO START ;;
;; load this file, echo-proximity, fa-i-calibration

(defun cb-proximity0 (proximity-msg)
  (if (boundp 'proximity0-init-value)
      (publish-FA-I-sensor-marker 0 (calc-arrow-size-in-rviz proximity-msg proximity0-init-value)))
  (setq proximity0-obj proximity-msg)
    (if (equal "T" (send proximity-msg :mode))
	(setq grasp-flag-front t)))

(defun cb-proximity1 (proximity-msg)
  (if (boundp 'proximity1-init-value)
      (publish-FA-I-sensor-marker 1 (calc-arrow-size-in-rviz proximity-msg proximity1-init-value)))
  (setq proximity1-obj proximity-msg))

(defun cb-proximity2 (proximity-msg)
  (if (boundp 'proximity2-init-value)
      (publish-FA-I-sensor-marker 2 (calc-arrow-size-in-rviz proximity-msg proximity2-init-value)))
  (setq proximity2-obj proximity-msg))

(defun cb-proximity3 (proximity-msg)
  (if (boundp 'proximity3-init-value)
      (publish-FA-I-sensor-marker 3 (calc-arrow-size-in-rviz proximity-msg proximity3-init-value)))
  (setq proximity3-obj proximity-msg)
  (if (equal "T" (send proximity-msg :mode))
	(setq grasp-flag-top t)))

(defun cb-proximity4 (proximity-msg)
  (if (boundp 'proximity4-init-value)
      (publish-FA-I-sensor-marker 4 (calc-arrow-size-in-rviz proximity-msg proximity4-init-value)))
  (setq proximity4-obj proximity-msg)
  (if (equal "T" (send proximity-msg :mode))
	(setq grasp-flag-top t)))

;; TODO END ;;

(defun echo-proximity()
    (ros::subscribe
     "/proximity_sensor0"
     force_proximity_ros::proximity #'cb-proximity0 10)
    (ros::subscribe
     "/proximity_sensor1"
     force_proximity_ros::proximity #'cb-proximity1 10)
    (ros::subscribe
     "/proximity_sensor2"
     force_proximity_ros::proximity #'cb-proximity2 10)
    (ros::subscribe
     "/proximity_sensor3"
     force_proximity_ros::proximity #'cb-proximity3 10)
    (ros::subscribe
     "/proximity_sensor4"
     force_proximity_ros::proximity #'cb-proximity4 10))


(defun grope-from-top()
  ;before use this function, do below
  ;(send *ri* :angle-vector (send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y (+ table-height 30)) :rpy #f(0 1.57 0))))

  ;;proximity1,2 are for width.
  ;;proximity0,3,4 is for depth.
  (echo-proximity)

  (let (;(proximity0-init-value) (proximity3-init-value) (proximity4-init-value)
	(proximity0-raw) (proximity3-raw) (proximity4-raw)
	(proximity0 0) (proximity3 0) (proximity4 0)
	(i 0) (width-sum 0))

    (FA-I-calibration)
    ;(ros::spin-once)
    ;;sensor calibration (far away from the target)
    ;; (setq proximity0-init-value (send proximity0-obj :average))
    ;; (setq proximity3-init-value (send proximity3-obj :average))
    ;; (setq proximity4-init-value (send proximity4-obj :average))

    (setq grasp-flag-top nil)
    (while (null grasp-flag-top)
      (setq proximity0-raw (send proximity0-obj :average))
      (setq proximity3-raw (send proximity3-obj :average))
      (setq proximity4-raw (send proximity4-obj :average))

      (setq proximity0 (- proximity0-raw proximity0-init-value))
      (setq proximity3 (- proximity3-raw proximity3-init-value))
      (setq proximity4 (- proximity4-raw proximity4-init-value))

      (format t "i : ~A~%" i)
      (format t "proximity0 : ~A~%" proximity0)
      (format t "proximity3 : ~A~%" proximity3)
      (format t "proximity4 : ~A~%" proximity4)


      (setq width-sum (+ width-sum (/ (- proximity4 proximity3) 100)))
      (setq i (- i 1))

      (send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector x y (+ table-height 30)) (float-vector 0 width-sum i)) :rpy #f(0 1.57 0)))
      (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 1000)

      (ros::spin-once))

    (send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector x y (+ table-height 30)) (float-vector 0 width-sum (+ i 20))) :rpy #f(0 1.57 0)))
    (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    (try-grasp)

    (send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y (+ table-height 30)) :rpy #f(0 1.57 0)))
    (send *ri* :angle-vector-raw (send *fetch* :angle-vector) 3000)))


;;try to pick object.
(defun pick()
  (send *ri* :angle-vector (send *fetch* :reset-pose))
  (send *ri* :stop-grasp)
  (send *ri* :wait-interpolation)
  (send msg :data "start")
  (ros::publish "stop_moveit" msg)


  (if (search-for-object)
      (progn
	(send msg :data "stop")
	(ros::publish "stop_moveit" msg)
	;(send *fetch* :inverse-kinematics (make-coords :pos (float-vector (- x 200) y (+ z 200))) :debug-view t)
	(send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector x y z) #f(-200 0 200)) :debug-view t))
	(move 10000)
	(let ((flag nil) (i 0))
	  (while (null flag)
	    ;(send *fetch* :inverse-kinematics (make-coords :pos (float-vector x y (+ z i))))
	    (send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector x y z) (float-vector -100 0 i))))
	    (setq flag (move 5000 1))
	    (setq i (+ i 5)))
	  (setq table-height (- i 5))
	(send *ri* :wait-interpolation)

	(grope-from-front)
	;; grope for the target by using proximity sensor(s) on fetch's hand
	;; (let ((proximity0 0) (threshold0 13000) (proximity-msg0)
	;;       (proximity1 0) (threshold1 13000) (proximity-msg1)
	;;       (j 0) (rend-y) (lend-y))
	;;   (while (< proximity0 threshold0)
	;;     (setq proximity-msg0
	;; 	  (one-shot-subscribe
	;; 	   "/proximity_sensor0"
	;; 	   force_proximity_ros::proximity))
	;;     (setq proximity0 (send proximity-msg0 :average))
	;;     (send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector x y z) (float-vector 0 j 0))))
	;;     (move 1000)
	;;     (setq j (+ j 5)))
	;;   (setq rend-y (+ y j))

	;;   (setq j 0)
	;;   (while (< proximity1 threshold1)
	;;     (setq proximity-msg1
	;; 	  (one-shot-subscribe
	;; 	   "/proximity_sensor1"
	;; 	   force_proximity_ros::proximity))
	;;     (setq proximity1 (send proximity-msg1 :average))
	;;     (send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector x y z) (float-vector 0 j 0))))
	;;     (move 1000)
	;;     (setq j (- j 5)))
	;;   (setq lend-y (+ y j))
	;;   (send *fetch* :inverse-kinematics (make-coords :pos (float-vector x (/ (+ rend-y lend-y) 2) z)))
	;;   (move 3000))

	;(send *fetch* :inverse-kinematics (make-coords :pos (float-vector (- x 200) y (+ z 200))))
	(send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector x y z) #f(-200 0 200))) :debug-view t)
	(move 1000)
	;(send *fetch* :inverse-kinematics (make-coords :pos (float-vector (- x 300) y (+ z 100))))
	(send *fetch* :inverse-kinematics (make-coords :pos (v+ (float-vector x y z) #f(-300 0 100))) :debug-view t)
	(move 2000)
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector 350 600 (+ z 100))) :debug-view t)
	(move 5000)
	(send *fetch* :inverse-kinematics (make-coords :pos (float-vector 350 600 100)) :debug-view t :rotation-axis nil)
	(move 5000)
	(send *ri* :stop-grasp)
	(init))
    (format t "trial canceled.~%"))))

;; elmer's glue: centroid too deep
;; fetch mistakes his hand for collision object.

