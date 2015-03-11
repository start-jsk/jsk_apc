#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")

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
    (ros::advertise-service "/semi/larm_move_for_verification"
                            jsk_2014_picking_challenge::ObjectVerification #'send self :cb-larm)
    (ros::advertise-service "/semi/rarm_move_for_verification"
                            jsk_2014_picking_challenge::ObjectVerification #'send self :cb-rarm)
    (ros::ros-info "subscriber initialized"))
  (:cb-larm (req) (send self :callback req :larm))
  (:cb-rarm (req) (send self :callback req :rarm))
  (:callback
    (req limb)
    (ros::ros-info "move_for_verification called")
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (let (avs untuck-coords verify-coords req-v res-v plist res)
      (push (send *baxter* :angle-vector) avs)  ;; set original av
      (setq untuck-coords (make-cascoords :pos (float-vector 550 (if (equal limb :larm) 200 -200) 1100)
                                          :rot #2f((-3.6e-06 0 -1) (0 1 0) (1 0 -3.6e-06))))
      (push (send *baxter* limb :inverse-kinematics untuck-coords
                  :rotation-axis :z :revert-if-fail nil) avs)
      (setq verify-coords (make-cascoords :pos (float-vector 650 (if (equal limb :larm) 80 -80) 1900)
                                          :rot #2f((-3.6e-06 0 -1) (0 1.0 0) (1 0 -3.6e-06))))
      (push (send *baxter* limb :inverse-kinematics verify-coords
                  :rotation-axis :x :revert-if-fail nil) avs)
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 8000))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; service call
      (ros::wait-for-service "/semi/sift_matcher")
      (setq req-v (instance jsk_2014_picking_challenge::ObjectMatchRequest :init))
      (send req-v :objects (send req :objects))
      (setq res-v (ros::service-call "/semi/sift_matcher" req-v))
      (setq plist (send res-v :probabilities))
      ;; return response
      (setq res (send req :response))
      (if (string-equal (elt (send req :objects) (maxindex plist)) (send req :target_object))
        (send res :succeeded t)
        (send res :succeeded nil))
      (send res :succeeded t)
      res)))

(ros::roseus "move_for_verification")
(setq m (instance move-for-verification-cb-class :init))
(ros::spin)

