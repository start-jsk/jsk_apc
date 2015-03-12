#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")

(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defun argmax (fvec)
  (let (ret)
    (dotimes (i (length fvec))
      (setq val (aref fvec i))
      (cond ((= i 0) (setq maxv val ret i))
            ((> val maxv) (setq maxv val ret i))
            (t nil)))
    ret))

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
    (ros::ros-info "[move_for_verification] initialized"))
  (:cb-larm (req) (send self :callback req :larm))
  (:cb-rarm (req) (send self :callback req :rarm))
  (:callback
    (req limb)
    (ros::ros-info "[move_for_verification] called")
    (let ((foldav-l #f(37.3096 -2.30713 -74.0698 146.667 99.5361 -3.66943 6.48193))
          (foldav-r #f(-48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745))
          (verifyav-l #f(12.7002 -19.9951 -106.765 127.463 73.4326 -85.4517 -25.0928))
          (verifyav-r #f(-11.6235 -21.9507 106.941 127.881 -63.2153 -90.0 13.5571))
          avs untuck-coords verify-coords req-v res-v plist res)
      ;; set original av
      (push (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
      ;; for limb which is not being used
      (if (equal limb :larm)
        (send *baxter* :rarm :angle-vector foldav-r)
        (send *baxter* :larm :angle-vector foldav-l))
      (push (send *baxter* :angle-vector) avs)
      ;; verify pose
      (if (equal limb :larm)
        (send *baxter* limb :angle-vector verifyav-l)
        (send *baxter* limb :angle-vector verifyav-r))
      (push (send *baxter* :angle-vector) avs)
      ;; reflect to *ri*
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 8000))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; service call
      (ros::ros-info "[move_for_verification] waiting for service /semi/sift_matcher")
      (ros::wait-for-service "/semi/sift_matcher")
      (ros::ros-info "[move_for_verification] found service /semi/sift_matcher")
      (setq req-v (instance jsk_2014_picking_challenge::ObjectMatchRequest :init))
      (ros::ros-info (format nil "[move_for_verification] objects: ~A~%" (send req :objects)))
      (send req-v :objects (send req :objects))
      (setq res-v (ros::service-call "/semi/sift_matcher" req-v))
      (setq plist (send res-v :probabilities))
      (ros::ros-info (format nil "[move_for_verification] probabilities: ~A~%" plist))
      ;; follow back same trajectory
      (send *ri* :angle-vector-sequence avs (make-list (length avs) :initial-element 8000))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; return response
      (setq res (send req :response))
      (when (= (length plist) 0) (ros::ros-info (format nil "Invalid probabilities: ~A" plist)))
      (if (string-equal (elt (send req :objects) (argmax plist)) (send req :target_object))
        (send res :succeeded t)
        (send res :succeeded nil))
      res)))

(ros::roseus "move_for_verification")
(setq m (instance move-for-verification-cb-class :init))
(ros::spin)

