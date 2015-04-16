#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/arm-motion.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")

(ros::load-ros-manifest "jsk_rviz_plugins")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defclass move-for-verification-cb-class
  :super propertied-object
  :slots ())
(defmethod move-for-verification-cb-class
  (:init
    ()
    (baxter-init)
    (send *baxter* :locate #f(0 0 950) :world)
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (pod-init)
    (orderbin-init)
    (objects (list *baxter* *pod* *orderbin*))
    (ros::advertise "/semi/master_status" jsk_rviz_plugins::OverlayText 1)
    (ros::advertise-service "/semi/larm_move_for_verification"
                            jsk_2014_picking_challenge::ObjectVerification #'send self :cb-larm)
    (ros::advertise-service "/semi/rarm_move_for_verification"
                            jsk_2014_picking_challenge::ObjectVerification #'send self :cb-rarm)
    (ros::ros-info "[move_for_verification] initialized"))
  (:cb-larm (req) (send self :callback req :larm))
  (:cb-rarm (req) (send self :callback req :rarm))
  (:callback
    (req arm)
    (let (avs plist msg res)
      (pushback (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
      ;; validate arm
      (ros::ros-info "validate arm")
      (unless (validate-arm arm)
        (send server :set-aborted (send server :result :sequence seq))
        (return-from :execute-cb nil))
      (setq avs (append avs (move-arm-for-verification arm)))
      (send *irtviewer* :draw-objects)
      ;; verify object
      (setq plist (send self :get-proba (send req :objects) "bof"))
      ; (setq plist (v+ (send self :get-proba (send req :objects) "sift")
      ;                 (send self :get-proba (send req :objects) "color_histogram")))
      (setq msg (instance jsk_rviz_plugins::OverlayText :init))
      (send msg :text (format nil "Objects: ~A~%Probabilities: ~A~%" (send req :objects) plist))
      (ros::publish "/semi/master_status" msg)
      ;; follow back same trajectory
      (send *ri* :angle-vector-sequence (reverse avs) :fast :default-controller 0 :scale 2.5)
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; return response
      (setq res (send req :response))
      (when (= (length plist) 0)
        (ros::ros-info (format nil "invalid probabilities: ~A" plist))
        (send res :succeeded nil)
        (return-from :callback res))
      (if (string-equal (elt (send req :objects) (argmax plist)) (send req :target_object))
        (send res :succeeded t)
        (send res :succeeded nil))
      res))
  (:get-proba
    (objects matcher)
    (let ((service-name (concatenate string "/semi/" matcher "_matcher")) req res)
      (ros::wait-for-service service-name)
      (setq req (instance jsk_2014_picking_challenge::ObjectMatchRequest :init))
      (send req :objects objects)
      (setq res (ros::service-call service-name req))
      (setq plist (send res :probabilities))
      plist)))

(ros::roseus "move_for_verification")
(instance move-for-verification-cb-class :init)
(ros::spin)

