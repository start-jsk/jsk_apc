;; vim: set ft=lisp:
;; -*- mode: lisp;-*-
(ros::load-ros-manifest "jsk_rviz_plugins")
(ros::load-ros-manifest "jsk_2015_05_baxter_apc")

(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/model/score-board.l")
(setq *overlay-text-width* 1200)
(setq *overlay-text-height* 50)

;; How to use
;;  (publish-bin-mark (list :a :b :c))
(defun publish-bin-mark(bin-list)
  (ros::advertise-if-yet "bin_marks" jsk_rviz_plugins::pictogramarray 5)
  (let ((bin-msg-list)
	(pict-array-msg (instance jsk_rviz_plugins::pictogramarray :init)))
    (dolist (bin bin-list)
      (let ((pos-vec (send *pod* bin))
	    (pose (instance geometry_msgs::Pose :init)))
	(send pose :position :x (/ (elt pos-vec 0) 1000))
	(send pose :position :y (/ (elt pos-vec 1) 1000))
	(send pose :position :z (- (/ (elt pos-vec 2) 1000) 0.95))
	(send pose :orientation :x 0)
	(send pose :orientation :y -0.7)
	(send pose :orientation :z 0)
	(send pose :orientation :w 0.7)
	(setq pict-msg (publish-popup-text (key-to-str bin) "base" "bin_marks" pose nil jsk_rviz_plugins::Pictogram::*ROTATE_X*))
	(push pict-msg bin-msg-list)
	))
    (send pict-array-msg :header :frame_id "base")
    (send pict-array-msg :header :stamp (ros::time-now))
    (send pict-array-msg :pictograms bin-msg-list)
    (ros::publish "bin_marks" pict-array-msg)
    )
  )

(defun publish-work-order-list(arm-str work-order)
  (let (msg object-array work-order-msg topic-name)
    (setq topic-name (format nil "/~A/work_order_list" arm-str))
    (ros::advertise-if-yet topic-name jsk_2015_05_baxter_apc::workorderarray 5)
    (setq msg (instance jsk_2015_05_baxter_apc::workorderarray :init))
    (setq object-array nil)
    (dolist (target work-order)
      (setq work-order-msg (instance jsk_2015_05_baxter_apc::workorder :init))
      (send work-order-msg :bin (key-to-str (str-to-key (elt target 0))))
      (send work-order-msg :object (key-to-str (str-to-key (elt target 1))))
      (push work-order-msg object-array)
      )
    (send msg :array object-array)
    (ros::publish topic-name msg)
    )
  )

(defun publish-popup-text(text frame topic_name pose &optional (publishp t) (action jsk_rviz_plugins::Pictogram::*JUMP*) (ttl 0.0) (size 0.7))
  (let ((pict-msg (instance jsk_rviz_plugins::Pictogram :init)))
    (when publishp (ros::advertise-if-yet topic_name jsk_rviz_plugins::pictogram 5))
    (send pict-msg :header :frame_id frame)
    (send pict-msg :header :stamp (ros::time-now))
    (send pict-msg :mode jsk_rviz_plugins::Pictogram::*STRING_MODE*)
    (send pict-msg :character text)
    (send pict-msg :ttl ttl)
    (send pict-msg :pose pose)
    (send pict-msg :size size)
    (send pict-msg :speed 0.3)
    (send pict-msg :color :r  (/ 25.0 255.0))
    (send pict-msg :color :g (/ 255.0 255.0))
    (send pict-msg :color :b (/ 240.0 255.0))
    (send pict-msg :color :a 1)
    (send pict-msg :action action)
    (if publishp
	(ros::publish topic_name pict-msg)
      pict-msg
      )))

(defun publish-end-coords ()
  (ros::advertise-if-yet "/larm_end_coords" visualization_msgs::Marker 5)
  (ros::advertise-if-yet "/rarm_end_coords" visualization_msgs::Marker 5)
  (ros::publish "/larm_end_coords" (eusobj->marker-msg (send (send (send *baxter* :larm :end-coords) :copy-worldcoords) :translate #f(0 0 -950) :world) (instance std_msgs::header :init :stamp (ros::time-now) :frame_id "/base")))
  (ros::publish "/rarm_end_coords"(eusobj->marker-msg (send (send (send *baxter* :rarm :end-coords) :copy-worldcoords) :translate #f(0 0 -950) :world) (instance std_msgs::header :init :stamp (ros::time-now) :frame_id "/base"))))

(defun publish-pod ()
  (let ((*simulator-origin* "base") pod-marker-msg)
    (if (not (boundp '*tfb*)) (setq *tfb* (instance ros::transform-broadcaster :init)))
    (send *tfb* :send-transform (send (make-coords) :translate #f(0 0 -950) :world) "base" "pod")
    (ros::advertise-if-yet "/pod_marker_array" visualization_msgs::MarkerArray 5)
    (unix::sleep 1)
    (setq pod-marker-msg (eusobj->marker-msg
			  *pod*
			  (instance std_msgs::header :init
				    :stamp (ros::time-now)
				    :frame_id "pod")
			  :ns "pod_body"
			  :lifetime 0 :id 2
			  :rainbow t))
    (send pod-marker-msg :color (vector->rgba (float-vector 0.4 0.4 0.4) 1))
    (ros::publish "/pod_marker_array"
                  (instance visualization_msgs::MarkerArray :init
                            :markers
                            (list (text->marker-msg
                                   "pod"
                                   (send (send *pod* :copy-worldcoords)
                                         :translate #f(-100 0 100))
                                   (instance std_msgs::header :init
                                             :stamp (ros::time-now)
                                             :frame_id "pod")
                                   :ns "pod_text" :color #f(1 0 0)
                                   :scale 100 :lifetime 0 :id 0)
                                  (coords->marker-msg
                                   (send *pod* :copy-worldcoords)
                                   (instance std_msgs::header :init
                                             :stamp (ros::time-now)
                                             :frame_id "pod")
                                   :ns "pod_coords"
                                   :lifetime 0 :id 1)
                                  pod-marker-msg
                                  )))))

(defun publish-marker-score-board ()
  (score-board-init)
  (ros::advertise-if-yet "/score_board" visualization_msgs::Marker 5)
  (if (not (boundp '*tfb*)) (setq *tfb* (instance ros::transform-broadcaster :init)))
  (send *tfb* :send-transform (send (make-coords) :translate #f(0 0 -950) :world) "base" "score_board")
  (unix::sleep 1)
  (ros::publish "/score_board"
                (object->marker-msg
                 *score-board*
                 (instance std_msgs::header :init
                           :stamp (ros::time-now)
                           :frame_id "score_board")
                 :ns "score_board_body"
                 :lifetime 0 :id 2)))

(defun publish-marker-score(score)
  (ros::advertise-if-yet "score_pict" jsk_rviz_plugins::pictogramarray 5)
  (let* ((bin-msg-list)
         (pict-array-msg (instance jsk_rviz_plugins::pictogramarray :init))
         (score-100 (round (/ score 100)))
         (score-10 (round (/ (mod score 100) 10)))
         (score-1 (round (mod score 10)))
         (offset 0))
    (dolist (ascore (list score-100 score-10 score-1))
      (let ((pose (instance geometry_msgs::Pose :init)))
        (setq offset (+ 0.6 offset))
        (send pose :position :x 5.0)
	(send pose :position :y ( - 0 (+  -1.2 offset) ))
        (send pose :position :z 2.6)
        (send pose :orientation :x 0)
        (send pose :orientation :y -0.7)
        (send pose :orientation :z 0)
        (send pose :orientation :w 0.7)
        (setq pict-msg (publish-popup-text (string ascore) "base" "score_pict" pose nil jsk_rviz_plugins::Pictogram::*ROTATE_X* 0 2.7))
        (push pict-msg bin-msg-list)
        ))
    (send pict-array-msg :header :frame_id "base")
    (send pict-array-msg :header :stamp (ros::time-now))
    (send pict-array-msg :pictograms bin-msg-list)
    (ros::publish "score_pict" pict-array-msg)
    )
  )


(defun publish-score (diff-score total)
  (let ((pose (instance geometry_msgs::pose :init)))
    (send pose :position :z 0.3)
    (send pose :orientation :y -0.7)
    (send pose :orientation :w 0.7)
    (publish-popup-text (format nil "+~AP" diff-score) "head" "apc_info" pose t jsk_rviz_plugins::Pictogram::*JUMP* 5.0 2.2)
    (publish-marker-score total)
    ))

(defun publish-overlay-text (text topic width height left top
				  &key
				  (text-size 12)
				  (line-width 2)
				  (font "DejaVu Sans Mono")
				  (fg-color (instance std_msgs::ColorRGBA :init :r  (/ 25.0 255.0) :g 1.0 :b (/ 240.0 255.0) :a 1.0))
				  (bg-color (instance std_msgs::ColorRGBA :init :r  0  :g 0  :b 0 :a 0.2)))
  (let ((overlay-text-msg (instance jsk_rviz_plugins::OverlayText :init)))
    (ros::advertise-if-yet topic jsk_rviz_plugins::OverlayText 5)
    (send overlay-text-msg :width width)
    (send overlay-text-msg :height height)
    (send overlay-text-msg :left left)
    (send overlay-text-msg :top top)
    (send overlay-text-msg :text_size text-size)
    (send overlay-text-msg :font font)
    (send overlay-text-msg :line_width 2)
    (send overlay-text-msg :text text)
    (send overlay-text-msg :fg_color fg-color)
    (send overlay-text-msg :bg_color bg-color)
    (ros::publish topic overlay-text-msg)
    )
  )

(defun publish-state-info (state arm)
  (publish-overlay-text (format nil "    ~A" state)
			(format nil "~A_state_info" (arm-to-str arm))
			*overlay-text-width* *overlay-text-height* 0
			(if (equal "left" (arm-to-str arm))  (* 1 *overlay-text-height*) (* 4 *overlay-text-height*) ):text-size 20))
