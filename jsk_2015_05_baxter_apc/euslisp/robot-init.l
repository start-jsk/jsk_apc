(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/model/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/model/order-bin.l")

(setq *simulator-p* nil)
(defun apc-init ()
  (baxter-init)
  (send *baxter* :locate #f(0 0 950) :world)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send (send *baxter* :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
  (send (send *baxter* :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
  (pod-init)
  (orderbin-init)
  (objects (list *baxter* *pod* *orderbin*))
  (setq *simulator-p* (not (send *ri* :joint-action-enable)))
  t)


;;For Debug
(defun publish-end-coords ()
  (ros::advertise "/larm_end_coords" visualization_msgs::Marker 5)
  (ros::advertise "/rarm_end_coords" visualization_msgs::Marker 5)
  (ros::publish "/larm_end_coords" (eusobj->marker-msg (send (send (send *baxter* :larm :end-coords) :copy-worldcoords) :translate #f(0 0 -950) :world) (instance std_msgs::header :init :stamp (ros::time-now) :frame_id "/base")))
  (ros::publish "/rarm_end_coords"(eusobj->marker-msg (send (send (send *baxter* :rarm :end-coords) :copy-worldcoords) :translate #f(0 0 -950) :world) (instance std_msgs::header :init :stamp (ros::time-now) :frame_id "/base"))))

(defun publish-pod ()
  (let ((*simulator-origin* "base"))
    (if (not (boundp '*tfb*)) (setq *tfb* (instance ros::transform-broadcaster :init)))
    (send *tfb* :send-transform (send *laundry* :worldcoords) *base-frame-id* "/pod")

    (ros::advertise "/object_detection_marker_array" visualization_msgs::MarkerArray 5)
    (unix::sleep 1)
    (ros::publish "/object_detection_marker_array"
		  (instance visualization_msgs::MarkerArray :init
			    :markers
			    (list (text->marker-msg
				   "pod"
				   (send (send *pod* :copy-worldcoords)
					 :translate #f(-100 0 100))
				   (instance std_msgs::header :init
					     :stamp (ros::time-now)
					     :frame_id "/pod")
				   :ns (format nil "object_text_~A" "pod") :color #f(1 0 0)
				   :scale 100 :lifetime 0 :id 0)
				  (coords->marker-msg
				   (send *pod* :copy-worldcoords)
				   (instance std_msgs::header :init
					     :stamp (ros::time-now)
					     :frame_id "/pod")
				   :ns (format nil "object_coords_~A" "pod")
				   :lifetime 0 :id 1)
				  (object->marker-msg
				   *pod*
				   (instance std_msgs::header :init
					     :stamp (ros::time-now)
					     :frame_id "/pod")
				   :ns (format nil "object_body_~A" "pod")
				   :lifetime 0 :id 2)
				  )))
    )