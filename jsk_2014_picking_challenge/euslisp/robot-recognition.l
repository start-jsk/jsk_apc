;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "jsk_recognition_msgs")
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
  (while
    (not (gethash arm *verify-object*))
    (unix::sleep 1)
    (ros::spin-once))
  (let* ((now (ros::time-now))
         (msg (gethash arm *verify-object*))
         (stamp (send (send msg :header) :stamp)))
    ;; wait for msg to synchronize
    (while
      (< (send stamp :sec) (send now :sec))
      (unix::sleep 1)
      (ros::spin-once)
      (setq msg (gethash arm *verify-object*))
      (setq stamp (send (send msg :header) :stamp))
      (setq correct? (string= target-object (send msg :matched)))
      ;; timeout
      (when (> (- (send (ros::time-now) :sec) (send now :sec)) timeout) (return)))
    correct?))

;; -------------------------------------
;; Check if grabbed with image_time_diff
;; -------------------------------------
(setq *image-time-diff* (make-hash-table))
(defun image-time-diff-cb (msg arm)
  (setf (gethash arm *image-time-diff*) msg))

(defun start-image-time-diff (arm)
  (let ((topic-ns (format nil "/see_~A_if_grabbed/image_time_diff" (arm-to-str arm))) msg)
    (ros::advertise-if-yet (format nil "~A/start" topic-ns) std_msgs::Header 1)
    (setq msg (instance std_msgs::Header :init))
    (send msg :stamp (ros::time-now))
    (ros::publish (format nil "~A/start" topic-ns) msg)
    ;; prepare for subscription
    (case arm
      (:larm (ros::subscribe (format nil "~A/output/diff" topic-ns)
                             jsk_recognition_msgs::ImageDifferenceValue
                             #'(lambda (msg) (image-time-diff-cb msg :larm))))
      (:rarm (ros::subscribe (format nil "~A/output/diff" topic-ns)
                             jsk_recognition_msgs::ImageDifferenceValue
                             #'(lambda (msg) (image-time-diff-cb msg :rarm)))))
    ))

(defparameter *image-time-diff-threshold* 0.17)
(defun see-if-grabbed (arm)
  (while
    (not (gethash arm *image-time-diff*))
    (unix::usleep 1000)
    (ros::spin-once))
  (let* ((now (ros::time-now))
         (msg (gethash arm *image-time-diff*))
         (stamp (send (send msg :header) :stamp)))
    ;; wait for msg to synchronize
    (while
      (< (send stamp :sec) (send now :sec))
      (unix::usleep 1000)
      (ros::spin-once)
      (setq msg (gethash arm *image-time-diff*))
      (setq stamp (send (send msg :header) :stamp))
      (setq diff (send msg :difference)))
    (if (> diff *image-time-diff-threshold*) t nil)))

;; -------------------------------------------------------
;; Check if grabbed with pressure sensor in vacuum cleaner
;; -------------------------------------------------------
(setq *gripper-grabbed* (make-hash-table))
(defun gripper-grabbed-cb (msg arm)
  (setf (gethash arm *gripper-grabbed*) (send msg :data)))

(ros::subscribe "/gripper_grabbed/limb/left/state" std_msgs::Bool
                #'(lambda (msg) (gripper-grabbed-cb msg :larm)))
(ros::subscribe "/gripper_grabbed/limb/right/state" std_msgs::Bool
                #'(lambda (msg) (gripper-grabbed-cb msg :rarm)))

(defun check-if-grabbed (arm)
  (ros::spin-once)
  (let (state)
    (setq state (gethash arm *gripper-grabbed*))
    ; (when state (speak-en "Grasped object" :google t))
    state))
