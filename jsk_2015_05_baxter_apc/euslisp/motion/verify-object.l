;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-visualization.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

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
