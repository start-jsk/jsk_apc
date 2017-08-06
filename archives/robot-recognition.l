;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_2015_05_baxter_apc")


;; -------------------------------------
;; Check if grabbed with image_time_diff
;; -------------------------------------
(if (not (boundp '*image-time-diff*))
    (setq *image-time-diff* (make-hash-table)))
(defun image-time-diff-cb (msg arm)
  (setf (gethash arm *image-time-diff*) msg))
(defun image-time-diff-larm-cb (msg) (image-time-diff-cb msg :larm))
(defun image-time-diff-rarm-cb (msg) (image-time-diff-cb msg :rarm))

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
                             #'image-time-diff-larm-cb))
      (:rarm (ros::subscribe (format nil "~A/output/diff" topic-ns)
                             jsk_recognition_msgs::ImageDifferenceValue
                             #'image-time-diff-rarm-cb)))
    ))

(defparameter *image-time-diff-threshold* 0.1)  ;; 0.03 : too sensitive
(defun see-if-grabbed (arm &key (timeout 10))
  (let* ((now (ros::time-now))
         (msg (gethash arm *image-time-diff*)))
    ;; wait for msg to synchronize
    (while
      (or (null msg) (< (send (send (send msg :header) :stamp) :sec) (send now :sec)))
      (unix::usleep 1000)
      (ros::spin-once)
      (setq msg (gethash arm *image-time-diff*))
      ;; timeout
      (when (> (- (send (ros::time-now) :sec) (send now :sec)) timeout)
        (ros::ros-error "timeout in see_if_grabbed ~A" (arm-to-str arm))
        (return-from see-if-grabbed nil)
        )
      )
    (unix::sleep 1)
    (ros::spin-once)
    (setq msg (gethash arm *image-time-diff*))
    (setq diff (send msg :difference))
    (ros::ros-info "see-if-grabbed ~A > ~A" diff *image-time-diff-threshold*)
    (if (> diff *image-time-diff-threshold*) t nil)
    ))

(defparameter *end-coords-diff-threshold-vertical* 65)
(defparameter *end-coords-diff-threshold-horizontal* 30)
(defun check-real-sim-end-coords-diff(arm pattern bin-objects)
  (let (real-coords simulator-coords moved-coords diff-threshold)
    (case pattern
          ((:left-most :right-most :left :right :middle :middle-left :middle-right)
           (setq diff-threshold *end-coords-diff-threshold-vertical*)
           )
          ((:rotate-and-left :rotate-and-right)
           (setq diff-threshold *end-coords-diff-threshold-horizontal*))
          )
    (send *ri* :update-robot-state)
    (send *baxter* arm :end-coords :draw-on :flush t :color #f(255 0 0))
    (send (*ri* . robot) arm :end-coords :draw-on :flush t :color #f(0 255 0))
    (setq real-coords (send (send (*ri* . robot) arm :root-link) :transformation (send (*ri* . robot) arm :end-coords)))
    (setq simulator-coords (send (send *baxter* arm :root-link) :transformation (send *baxter* arm :end-coords)))
    (setq moved-coords (send real-coords :transformation simulator-coords  :local))
    (ros::ros-info "check-rel-sim-end-coords-diff diff:~A thre:~A" (elt (send moved-coords :pos) 2) diff-threshold)
    (if (> (elt (send moved-coords :pos) 2) diff-threshold)
	(progn
	  (ros::ros-error "real and sim's end-coords diff is over threshold!!")
	  nil
	  )
      t)))

(defun check-if-grabbed (arm pattern bin-objects)
  (ros::spin-once)
  (let (state)
    (unless (check-real-sim-end-coords-diff arm pattern bin-objects)
      (ros::ros-info "check real sim end coords")
      (return-from check-if-grabbed nil)
      )
    (setq state (gethash arm *gripper-grabbed*))
    ; (when state (speak-en "Grasped object" :google t))
    state))

;; ---------------------------------
;; Detect Object Position in the bin
;; ---------------------------------
(setq *detected-labels* (make-hash-table))
(defun detect-object-pos-cb (msg arm) (setf (gethash arm *detected-labels*) (send msg :labels)))
(defun detect-object-pos-larm-cb (msg) (detect-object-pos-cb msg :larm))
(defun detect-object-pos-rarm-cb (msg) (detect-object-pos-cb msg :rarm))
(ros::subscribe "/left_object_detection/feature_label_object_detector/output"
                jsk_2015_05_baxter_apc::ProbabilisticLabels
                #'detect-object-pos-larm-cb)
(ros::subscribe "/right_object_detection/feature_label_object_detector/output"
                jsk_2015_05_baxter_apc::ProbabilisticLabels
                #'detect-object-pos-rarm-cb)
(defun detect-object-pos-in-bin (arm)
  (unix::sleep 1)
  (ros::spin-once)
  (gethash arm *detected-labels*)
  )



;;--------------------------------
;; Bounding Box Array in the bin
;;--------------------------------
(setq *bounding-box-array-in-bin* (make-hash-table))
(setq *now-bin-id* nil)
(defun bounding-box-array-in-bin-cb (msg)
  (ros::ros-info "~A" *now-bin-id*)
  (setf (gethash *now-bin-id* *bounding-box-array-in-bin*) msg))

(defun bounding-box-array-in-bin ( &key (timeout 20))
  (let* ((now (ros::time-now))
         (msg nil))
    (dolist (bin (list :a :b :c :d :e :f :g :h :i :j :k :l))
      (setq now (ros::time-now))
      (setq *now-bin-id* bin)
      (ros::ros-info "wait for bounding_box_array msg: ~A" (key-to-str bin))
      (ros::subscribe (format nil "/extract_~A_indices/box_array" (key-to-str bin))
		      jsk_recognition_msgs::BoundingBoxArray
		      #'bounding-box-array-in-bin-cb)
      (setq msg nil)
      
      (while  ;; wait for msg to synchronize
	  (or (null msg))
	(unix::sleep 1)
	(ros::spin-once)
	(setq msg (gethash bin *bounding-box-array-in-bin*))
	;; timeout
	(when (> (- (send (ros::time-now) :sec) (send now :sec)) timeout)
	  (ros::ros-error "timeout in bounding_box_array: ~A" (key-to-str bin))
	  (return)))
      (ros::unsubscribe (format nil "/extract_~A_indices/box_array" (key-to-str bin)))
      )
    *bounding-box-array-in-bin*))
