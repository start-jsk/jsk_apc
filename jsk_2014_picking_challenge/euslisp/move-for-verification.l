#! /usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")

(ros::roseus "move_for_verification")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

()

(defclass move-for-verification-cb-class
  :super propertied-object
  :slots ())

(defmethod move-for-verification-cb-class
  (:init ()
	 (baxter-init)
	 (send *baxter* :locate #f(0 0 950) :world)
	 (objects (list *baxter*))
	 (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
	 (unix::sleep 3)
	 (ros::advertise-service "/semi/larm_move_for_verification" jsk_2014_picking_challenge::ObjectVerification #'send self :cb-larm)
	 (ros::advertise-service "/semi/rarm_move_for_verification" jsk_2014_picking_challenge::ObjectVerification #'send self :cb-rarm)
	 (ros::ros-info "subscriber initialized"))
  (:cb-larm (req)
	    (send self :callback req ':larm))
  (:cb-rarm (req)
	    (send self :callback req ':rarm))
  (:callback (req arm)
	     (ros::ros-info "move_for_verification called")
	     (let ()
	       ;;display
	       (objects (list *baxter*))
	       ;;move
	       (setq target (instance cascaded-coords :init :pos #f(700 0 1200)))
	       (when (not (send *baxter* arm :inverse-kinematics target :rotate-axis :z :debug-view :no-message))
		 (send *baxter* :reset-pose)
		 (send *baxter* arm :inverse-kinematics target :rotate-axis :z :debug-view :no-message))
	       (send *ri* :angle-vector (send *baxter* :angle-vector))
	       (send *irtviewer* :draw-objects)
	       (x::window-main-one)
	       ;;service call
	       (ros::wait-for-service "/semi/sift_matcher")
	       (setq req-v (instance roseus::ObjectMatchRequest :init))
	       (setq req-v :objects (send req :objects))
	       (setq res-v (ros::service-call "/semi/sift_matcher" req-v))
	       (setq plist (send res-v :probabilities))
	       ;;return response
	       (setq res (send req :response))
	       (setq i (position (apply #'max plist) plist)) ;index of max
	       (if (string-equal (elt (send req :objects) i) (send req :target_object)) (send res :succeeded t) (send res :succeeded nil))
	       res)))

(setq m (instance move-for-verification-cb-class :init))
(ros::spin)
