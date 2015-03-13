#! /usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/gripper-control.l")

(ros::roseus "put_orderbin")
;(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defclass put-orderbin-cb-class
  :super propertied-object
  :slots ())

(defmethod put-orderbin-cb-class
  (:init ()
	 (baxter-init)
	 (send *baxter* :locate #f(0 0 950) :world)
	 (objects (list *baxter*))
	 (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
	 (unix::sleep 3)
	 (ros::advertise-service "/semi/larm_put_orderbin" jsk_2014_picking_challenge::ReleaseItem #'send self :cb-larm)
	 (ros::advertise-service "/semi/rarm_put_orderbin" jsk_2014_picking_challenge::ReleaseItem #'send self :cb-rarm)
	 (ros::ros-info "subscriber initialized"))
  (:cb-larm (req)
	    (send self :callback req ':larm))
  (:cb-rarm (req)
	    (send self :callback req ':rarm))
  (:callback (req arm)
	     (ros::ros-info "put-orderbin called")
	     (let ()
	       ;;prepare pod-lowres
	       (when (not (boundp '*pod*))
		 (load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
		 (send *pod* :locate #f(1300 0 0) :world)
		 (send *pod* :rotate pi/2 :x :world)
		 (send *pod* :rotate (- pi/2) :z :world)
		 )
	       ;;prepare order-bin
	       (when (not (boundp '*order-bin*))
		 (load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")
		 (send *order-bin* :locate #f(650 0 500) :world) ;put order-bin at the center between baxter and kiva-pod
		 (send *order-bin* :rotate pi/2 :z :world)
		 )
	       ;;display
	       (objects (list *baxter* *pod* *order-bin*))
	       ;;move item to orderbin
	       (setq target (instance cascaded-coords :init :pos #f(650 0 750)))
	       (when (not (send *baxter* arm :inverse-kinematics target :rotate-axis :z :debug-view :no-message))
		 (send *baxter* :reset-pose)
		 (send *baxter* arm :inverse-kinematics target :rotate-axis :z :debug-view :no-message))
	       ;;(send *baxter* :look-at-hand :larm)
	       (send *ri* :angle-vector (send *baxter* :angle-vector))
	       (send *ri* :stop-grasp)
	       (send *baxter* :reset-pose)
	       (send *ri* :angle-vector (send *baxter* :angle-vector))
	       (send *irtviewer* :draw-objects)
	       (x::window-main-one)
	       ;;return response
	       (setq res (send req :response))
	       (send res :succeeded t)
	       res
	       )))

(setq m (instance put-orderbin-cb-class :init))
(ros::spin)
