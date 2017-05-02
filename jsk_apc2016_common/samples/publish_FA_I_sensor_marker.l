#!/usr/bin/env roseus

(ros::roseus "publish_FA_I_sensor_marker")

(defun make-arrow (size)
  (print size)
  (let ((radius (max 1.0 size))(cylinder) (cone) (arrow))
    (setq cylinder (make-cylinder (/ radius 2) (* radius 4)))
    (setq cone (make-cone (float-vector 0 0 (* radius 2)) radius))
    (send cone :translate (float-vector 0 0 (* radius 4)))
    (send cylinder :assoc cone)
    (setq arrow (body+ cone cylinder))
    arrow))


(ros::advertise "/arrow_marker0" visualization_msgs::Marker 5)
(ros::advertise "/arrow_marker1" visualization_msgs::Marker 5)
(ros::advertise "/arrow_marker2" visualization_msgs::Marker 5)
(ros::advertise "/arrow_marker3" visualization_msgs::Marker 5)
(ros::advertise "/arrow_marker4" visualization_msgs::Marker 5)

(defun publish-FA-I-sensor-marker (&optional (num 0) (size 10))
  (let ((arrow (make-arrow size))
	(pub-arrow-topic (format nil "/arrow_marker~A" num))
	(gripper))
    (ros::rate 10)
    (send arrow :set-color :red)

    (cond
    ((eq num 0)
     (setq gripper (make-coords))
     (send gripper :locate #f(-30 0 0))
     (send gripper :rotate (deg2rad 90) :y))
    ((eq num 1)
     (setq gripper (make-coords))
     (send gripper :locate #f(0 -50 20))
     (send gripper :rotate (deg2rad -90) :x))
    ((eq num 2)
     (setq gripper (make-coords))
     (send gripper :locate #f(0 50 -20))
     (send gripper :rotate (deg2rad 90) :x))
    ((eq num 3)
     (setq gripper (make-coords))
     (send gripper :locate #f(30 -60 0))
     (send gripper :rotate (deg2rad 90) :y))
    ((eq num 4)
     (setq gripper (make-coords))
     (send gripper :locate #f(30 60 0))
     (send gripper :rotate (deg2rad 90) :y))
    (t (format t "arg error!~%")))

    (let* ((header (instance std_msgs::header :init
			     :stamp (ros::time-now)
			     :frame_id "/gripper_link"))
	   (arrow-msg (object->marker-msg arrow header :coords gripper)))

      (ros::publish pub-arrow-topic arrow-msg)

      ;; for array msg
      ;; you shold set up namespace (:ns ) or id (:id) for marker array
      (send arrow-msg :ns "test_arrow")
      ;;(print header)
      )))

;(publish-FA-I-sensor-marker)
;(make-arrow 300)
;(make-sensor-coords)
