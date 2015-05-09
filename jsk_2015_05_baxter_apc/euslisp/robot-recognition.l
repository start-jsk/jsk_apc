;; vim: set ft=lisp:
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "std_msgs")

;; -------------------------------------
;; Check if grabbed with image_time_diff
;; -------------------------------------
(setq *image-time-diff* (make-hash-table))
(defun image-time-diff-cb (msg namespace)
  (setf (gethash (intern (string-upcase namespace)) *image-time-diff*) (send msg :data)))

(defun start-image-time-diff (&key namespace)
  (let (msg)
    (unless (ros::get-topic-publisher "image_time_diff/start_comparing")
      (ros::advertise "image_time_diff/start_comparing" std_msgs::String 1)
      (unix::sleep 1))
    (setq msg (instance std_msgs::String :init))
    (send msg :data namespace)
    (ros::publish "image_time_diff/start_comparing" msg)
    (unless (ros::get-topic-subscriber (format nil "image_time_diff/output/~A" namespace))
      (ros::subscribe (format nil "image_time_diff/output/~A/diff" namespace) std_msgs::Float32
                      #'(lambda (msg) (image-time-diff-cb msg "right_hand")))
      (unix::sleep 1)
      )
    ))

(defparameter *image-time-diff-threshold* 30)
(defun see-if-grabbed (&key namespace)
  (let (diff)
    (ros::spin-once)
    (setq diff (gethash (intern (string-upcase namespace)) *image-time-diff*))
    (if (> diff *image-time-diff-threshold*) t nil)
    ))
