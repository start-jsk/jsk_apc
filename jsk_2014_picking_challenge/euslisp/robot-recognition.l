;; vim: set ft=lisp:

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

;; -------------
;; Verify object
;; -------------
(setq *verify-object* (make-hash-table))
(defun verify-object-cb (msg arm) (setf (gethash arm *verify-object*) msg))
(ros::subscribe "left_process/object_verification/output"
                jsk_2014_picking_challenge::ObjectRecognition
                #'(lambda (msg) (verify-object-cb msg :larm)))
(ros::subscribe "right_process/object_verification/output"
                jsk_2014_picking_challenge::ObjectRecognition
                #'(lambda (msg) (verify-object-cb msg :rarm)))

(defun verify-object (arm target-object &key (timeout 20))
  (ros::spin-once)
  (let ((now (ros::time-now))
        (stamp (send (send (gethash arm *verify-object*) :header) :stamp)))
    (while
      (< (send stamp :sec) (send now :sec))
      (unix::sleep 1)
      (ros::spin-once)
      (setq correct? (string= target-object (send verify-object-msg :matched)))
      (when (> (- (send (ros::time-now) :sec) (send now :sec)) timeout) (return)))
    correct?))

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

;; -------------------------------------------------------
;; Check if grabbed with pressure sensor in vacuum cleaner
;; -------------------------------------------------------
(setq *gripper-grabbed* (make-hash-table))
(defun gripper-grabbed-cb (msg arm)
  (setf (gethash arm *gripper-grabbed*) (send msg :data)))

(ros::subscribe "gripper_grabbed/limb/left/state" std_msgs::Bool
                #'(lambda (msg) (gripper-grabbed-cb msg :larm)))
(ros::subscribe "gripper_grabbed/limb/right/state" std_msgs::Bool
                #'(lambda (msg) (gripper-grabbed-cb msg :rarm)))

(defun check-if-grabbed (arm)
  (ros::spin-once)
  (let (state)
    (setq state (gethash arm *gripper-grabbed*))
    ; (when state (speak-en "Grasped object" :google t))
    state)
