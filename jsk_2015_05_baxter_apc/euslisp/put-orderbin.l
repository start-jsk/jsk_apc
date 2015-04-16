#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/arm-motion.l")
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
    (ros::ros-info "[put_orderbin] initialized"))
  (:cb-larm (req) (send self :callback req :larm))
  (:cb-rarm (req) (send self :callback req :rarm))
  (:callback
    (req arm)
    (let (avs res)
      (ros::ros-info "[put_orderbin] called")
      (pushback (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
      (setq avs (append avs (move-to-order-bin arm)))
      (send *irtviewer* :draw-objects)
      (send *ri* :stop-grasp)  ;; release object
      ;; follow back same trajectory
      (send *ri* :angle-vector-sequence (reverse avs))
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
(setq r (ros::rate 10))
(do-until-key (ros::ok)
  (send *ri* :spin-once) ;; (ros::spin-once "baxter_interface")
  (ros::spin-once)
  (ros::sleep))

