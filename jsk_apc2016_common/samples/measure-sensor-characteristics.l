#!/usr/bin/env roseus

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "geometry_msgs")
(ros::load-ros-manifest "sensor_msgs")
(ros::load-ros-manifest "roseus")
(ros::load-ros-package "force_proximity_ros")

(require "package://fetcheus/fetch-interface.l")

(fetch-init t)

(load "package://jsk_apc2016_common/samples/publish_FA_I_sensor_marker.l")

;; initialize pose of the real fetch robot
(defun init()
      (send *fetch* :reset-pose)
      (send *fetch* :head :neck-p :joint-angle 15) ;; see down
      (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
      (send *ri* :stop-grasp)
      (send *ri* :wait-interpolation))

(defun arm-set (&optional (torso 235.876) (time 5000))
  ;; pos #f(800 0 770) rpy #f(0 1.57 0)
  (send *fetch* :angle-vector (float-vector torso 0.007561 9.1053 -179.963 28.8795 -180.039 109.729 -0.011606 0.0 0.0))
  (send *ri* :angle-vector (send *fetch* :angle-vector) time)
  (send *ri* :wait-interpolation))

;; order the real fetch robot to follow the joint angle of the model robot
(defun move (time &optional (trial-number 100))
  (let ((test nil) (i 0))
    (while (and (null test) (< i trial-number))
      (setq i (+ i 1))
      (setq test (send *ri* :angle-vector (send *fetch* :angle-vector) time)))
    (send *ri* :wait-interpolation)))


(defun cb-proximity (proximity-msg)
  (setq proximity-sum (+ proximity-sum (send proximity-msg :proximity)))
  (setq count-spin (+ count-spin 1)))


(defun echo-proximity()
    (ros::subscribe
     "/proximity_sensor0"
     force_proximity_ros::proximity #'cb-proximity 100))

(defun write-to-file(val1 val2)
  (with-open-file (out "FA_I_sensor.dat"
		       :direction :output
		       :if-exists :append
		       :if-does-not-exist :create)
		  (princ (format nil "~A ~A~%" val1 val2) out)))

;;be careful when use this function
;;table-height ~ 770
(defun measure ()
  (echo-proximity)
  (arm-set 235.876 5000)
  (dotimes (i 50)
    (unix:sleep 1)
    (setq proximity-sum 0)
    (setq count-spin 0)
    (ros::spin-once)
    (while (< count-spin 100)
      ;(print count-spin)
      )
    (setq average-proximity (/ (float proximity-sum) (float count-spin)))
    (write-to-file i average-proximity)
    (format t "i : ~A~%" i)
    (send *fetch* :torso :waist-z :joint-angle (+ 235.876 i))
    (send *ri* :angle-vector-raw (send *fetch* :angle-vector))
    (send *ri* :wait-interpolation)
    ))


(defun measure-closing()
  (echo-proximity)
  (arm-set 285.876 5000)
  (dotimes (i 50)
    (unix:sleep 1)
    (setq proximity-sum 0)
    (setq count-spin 0)
    (ros::spin-once)
    (while (< count-spin 100)
      ;;(print count-spin)
      )
    (setq average-proximity (/ (float proximity-sum) (float count-spin)))
    (write-to-file (- 50 i) average-proximity)
    (format t "50 - i : ~A~%" (- 50 i))
    ;;(send *fetch* :torso :waist-z :joint-angle (+ 235.876 i))
    (send *fetch* :torso :waist-z :joint-angle (- 285.876 i))
    (send *ri* :angle-vector-raw (send *fetch* :angle-vector))
    (send *ri* :wait-interpolation)
    ))
