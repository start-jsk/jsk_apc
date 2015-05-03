(ros::load-ros-manifest "jsk_rviz_plugins")
(load "package://jsk_2014_picking_challenge/euslisp/model/score-board.l")
(setq *score* 0)

;; How to use
;;  (publish-bin-mark (list :a :b :c))
(defun publish-bin-mark(bin-list)
  (unless (ros::get-topic-publisher "bin_marks")
    (ros::advertise "bin_marks" jsk_rviz_plugins::pictogramarray 5)
    (unix:sleep 1))
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
	(setq pict-msg (publish-popup-text (symbol-to-str bin) "base" "bin_marks" pose nil jsk_rviz_plugins::Pictogram::*ROTATE_X*))
	(push pict-msg bin-msg-list)
	))
    (send pict-array-msg :header :frame_id "base")
    (send pict-array-msg :header :stamp (ros::time-now))
    (send pict-array-msg :pictograms bin-msg-list)
    (ros::publish "bin_marks" pict-array-msg)
    )
  )

(defun publish-popup-text(text frame topic_name pose &optional (publishp t) (action jsk_rviz_plugins::Pictogram::*JUMP*) (ttl 0.0))
  (let ((pict-msg (instance jsk_rviz_plugins::Pictogram :init)))
    (if publishp
	(unless (ros::get-topic-publisher topic_name)
	  (ros::advertise topic_name jsk_rviz_plugins::pictogram 5)
	  (unix:sleep 1)))
    (send pict-msg :header :frame_id frame)
    (send pict-msg :header :stamp (ros::time-now))
    (send pict-msg :mode jsk_rviz_plugins::Pictogram::*STRING_MODE*)
    (send pict-msg :character text)
    (send pict-msg :ttl ttl)
    (send pict-msg :pose pose)
    (send pict-msg :size 0.7)
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
  (ros::advertise "/larm_end_coords" visualization_msgs::Marker 5)
  (ros::advertise "/rarm_end_coords" visualization_msgs::Marker 5)
  (ros::publish "/larm_end_coords" (eusobj->marker-msg (send (send (send *baxter* :larm :end-coords) :copy-worldcoords) :translate #f(0 0 -950) :world) (instance std_msgs::header :init :stamp (ros::time-now) :frame_id "/base")))
  (ros::publish "/rarm_end_coords"(eusobj->marker-msg (send (send (send *baxter* :rarm :end-coords) :copy-worldcoords) :translate #f(0 0 -950) :world) (instance std_msgs::header :init :stamp (ros::time-now) :frame_id "/base"))))

(defun publish-pod ()
  (let ((*simulator-origin* "base"))
    (if (not (boundp '*tfb*)) (setq *tfb* (instance ros::transform-broadcaster :init)))
    (send *tfb* :send-transform (send (make-coords) :translate #f(0 0 -950) :world) "base" "pod")
    (ros::advertise "/pod_marker_array" visualization_msgs::MarkerArray 5)
    (unix::sleep 1)
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
                                  (eusobj->marker-msg
                                   *pod*
                                   (instance std_msgs::header :init
                                             :stamp (ros::time-now)
                                             :frame_id "pod")
                                   :ns "pod_body"
                                   :lifetime 0 :id 2
                                   :rainbow t)
                                  )))))

(defun publish-score-board ()
  (score-board-init)
  (ros::advertise "/score_board" visualization_msgs::Marker 5)
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

(defun publish-score(score)
  (unless (ros::get-topic-publisher "score_pict")
    (ros::advertise "score_pict" jsk_rviz_plugins::pictogramarray 5)
    (unix:sleep 1))
  (if (not (boundp '*tfb*)) (setq *tfb* (instance ros::transform-broadcaster :init)))
  (send *tfb* :send-transform (send (make-coords) :translate #f(0 0 -950) :world) "base" "score_board")
  (let* ((bin-msg-list)
         (pict-array-msg (instance jsk_rviz_plugins::pictogramarray :init))
         (score-100 (/ score 100))
         (score-10 (/ (mod score 100) 10))
         (score-1 (mod score 10))
         (offset 0))
    (dolist (ascore (list score-100 score-10 score-1))
      (let ((pose (instance geometry_msgs::Pose :init)))
        (setq offset (+ 0.15 offset))
        (send pose :position :x (+  -0.35 offset))
        (send pose :position :y 2.0)
        (send pose :position :z -0.25)
        (send pose :orientation :x 0)
        (send pose :orientation :y -0.7)
        (send pose :orientation :z 0)
        (send pose :orientation :w 0.7)
        (setq pict-msg (publish-popup-text (string ascore) "base" "score_pict" pose nil jsk_rviz_plugins::Pictogram::*ROTATE_X*))
        (push pict-msg bin-msg-list)
        ))


    (let ((offset 0) (score-board-panel-pose (instance geometry_msgs::Pose :init)))
      ;;score board
      (setq offset (+ 0.15 offset))
      (send score-board-panel-pose :position :x 0)
      (send score-board-panel-pose :position :y 1.78)
      (send score-board-panel-pose :position :z -0.7)
      (send score-board-panel-pose :orientation :x 0.5)
      (send score-board-panel-pose :orientation :y -0.5)
      (send score-board-panel-pose :orientation :z 0.5)
      (send score-board-panel-pose :orientation :w 0.5)
      (setq pict-msg (publish-popup-text "Score Board" "base" "score_pict" score-board-panel-pose nil jsk_rviz_plugins::Pictogram::*JUMP_ONCE*))
      (push pict-msg bin-msg-list)
      ;;score board end
      )

   (let ((offset 0) (score-p-pose (instance geometry_msgs::Pose :init)))
      ;;score board
      (setq offset (+ 0.15 offset))
      (send score-p-pose :position :x 0.25)
      (send score-p-pose :position :y 1.78)
      (send score-p-pose :position :z -0.2)
      (send score-p-pose :orientation :x 0.5)
      (send score-p-pose :orientation :y -0.5)
      (send score-p-pose :orientation :z 0.5)
      (send score-p-pose :orientation :w 0.5)
      (setq pict-msg (publish-popup-text "P" "base" "score_pict" score-p-pose nil jsk_rviz_plugins::Pictogram::*JUMP_ONCE*))
      (push pict-msg bin-msg-list)
      ;;score board end
      )

    (send pict-array-msg :header :frame_id "base")
    (send pict-array-msg :header :stamp (ros::time-now))
    (send pict-array-msg :pictograms bin-msg-list)
    (ros::publish "score_pict" pict-array-msg)
    )
  )


(defun add-score (diff-score)
  (let ((pose (instance geometry_msgs::pose :init)))
    (send pose :position :z 0.3)
    (send pose :orientation :y -0.7)
    (send pose :orientation :w 0.7)
    (publish-popup-text (format nil "+~AP" diff-score) "head" "apc_info" pose t jsk_rviz_plugins::Pictogram::*JUMP* 5.0)
    (setq *score* (+ diff-score *score*))
    (publish-score-board)
    (publish-score *score*)
    ))
