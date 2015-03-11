#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")
(load "package://jsk_2014_picking_challenge/euslisp/gripper-control.l")

(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defclass put-orderbin-cb-class
  :super propertied-object
  :slots ())
(defmethod put-orderbin-cb-class
  (:init
    ()
    (baxter-init)
    (send *baxter* :locate #f(0 0 950) :world)
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (pod-init)
    (orderbin-init)
    (objects (list *baxter* *pod* *orderbin*))
    (ros::advertise-service "/semi/larm_put_orderbin" jsk_2014_picking_challenge::ReleaseItem #'send self :cb-larm)
    (ros::advertise-service "/semi/rarm_put_orderbin" jsk_2014_picking_challenge::ReleaseItem #'send self :cb-rarm)
    (ros::ros-info "subscriber initialized"))
  (:cb-larm (req) (send self :callback req :larm))
  (:cb-rarm (req) (send self :callback req :rarm))
  (:callback
    (req limb)
    (ros::ros-info "put-orderbin called")
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (let ()
      ;; move object to orderbin
      (setq avs nil)
      (push (send *baxter* :angle-vector) avs)
      (setq untuck-coords (make-cascoords :pos (float-vector 550 (if (equal limb :larm) 200 -200) 1100)
                                          :rot #2f((-3.6e-06 0 -1) (0 1 0) (1 0 -3.6e-06))))
      (push (send *baxter* limb :inverse-kinematics untuck-coords
                  :rotation-axis :z :revert-if-fail nil) avs)
      (setq target (make-cascoords :pos (send *orderbin* :transform-vector #f(0 0 300))
                                   :rot #2f((-3.6e-06 0 -1) (0 1 0) (1 0 -3.6e-06))))
      (push (send *baxter* limb :inverse-kinematics target :rotation-axis :z :revert-if-fail nil) avs)
      (push (send *baxter* limb :move-end-pos #f(0 0 -250) :world :revert-if-fail nil) avs)
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 8000))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      (send *ri* :stop-grasp)  ;; release object
      ;; follow back same trajectory
      (send *ri* :angle-vector-sequence avs (make-list (length avs) :initial-element 8000))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;;return response
      (setq res (send req :response))
      (send res :succeeded t)
      res))
  )

(ros::roseus "put_orderbin")
(ros::rate 500)
(setq m (instance put-orderbin-cb-class :init))
(ros::spin)

