#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")

(ros::load-ros-manifest "jsk_rviz_plugins")
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
    (ros::advertise "/semi/master_status" jsk_rviz_plugins::OverlayText 1)
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
    (let (
          ; (foldav-l #f(37.3096 -2.30713 -74.0698 146.667 99.5361 -3.66943 6.48193))
          ; (foldav-r #f(-48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745))
          (foldav-r-upper #f(7.80029 -1.12061 82.9248 141.438 -116.477 -6.48193 -8.10791))
          (foldav-l-upper #f(-24.1919 1.07666 -90.9009 131.572 -58.4912 -0.812988 54.7559))
          (verifyav-l #f(1.56006 -10.6348 -130.254 83.584 72.29 -15.4028 157.192))
          (verifyav-r #f(-22.0825 15.1392 144.448 111.006 -25.0488 2.54883 24.895))
          avs untuck-coords verify-coords req-v res-v plist res msg)
      ;; set original av
      (push (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
      ;; to avoid shelf for using arm
      (case limb
        (:larm (send *baxter* limb :angle-vector foldav-l-upper))
        (:rarm (send *baxter* limb :angle-vector foldav-r-upper)))
      (push (send *baxter* :angle-vector) avs)
      ; (case limb
      ;   (:larm (send *baxter* limb :angle-vector foldav-l))
      ;   (:rarm (send *baxter* limb :angle-vector foldav-r)))
      ; (push (send *baxter* :angle-vector) avs)
      ;; verify pose
      (case limb
        (:larm (send *baxter* limb :angle-vector verifyav-l))
        (:rarm (send *baxter* limb :angle-vector verifyav-r)))
      (push (send *baxter* :angle-vector) avs)
      ;; reflect to *ri*
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 1))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; service call
      (unix::sleep 10)

      ;;(setq plist (send self :get-probabilities (send req :objects) "sift"))
      (setq plist (v+ (send self :get-probabilities (send req :objects) "sift")
                      (send self :get-probabilities (send req :objects) "color_histogram")))

      (setq msg (instance jsk_rviz_plugins::OverlayText :init))
      (send msg :text (format nil "Objects: ~A~%Probabilities: ~A~%"
                              (send req :objects) plist))
      (ros::publish "/semi/master_status" msg)


      (ros::ros-info (format nil "[move_for_verification] probabilities: ~A~%" plist))
      ;; follow back same trajectory
      (setq avs (reverse avs))
      (pop avs)
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 1))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; return response
      (setq res (send req :response))
      (when (= (length plist) 0)
        (ros::ros-info (format nil "Invalid probabilities: ~A" plist))
        (send res :succeeded nil)
        (return-from :callback res))
      (if (string-equal (elt (send req :objects) (argmax plist)) (send req :target_object))
        (send res :succeeded t)
        (send res :succeeded nil))
      res))
    (:get-probabilities
   (objects matcher-name)
   (let ((service-name (concatenate string "/semi/" matcher-name "_matcher")))
     (ros::ros-info (format nil "[move_for_verification] waiting for service /semi/~A_matcher" matcher-name))
     (ros::wait-for-service service-name)
     (ros::ros-info (format nil "[move_for_verification] found service /semi/~A_matcher" matcher-name))
     (setq req-v (instance jsk_2014_picking_challenge::ObjectMatchRequest :init))
     (ros::ros-info (format nil "objects: ~A~%" objects))
     (send req-v :objects objects)
     (setq res-v (ros::service-call service-name req-v))
     (setq plist (send res-v :probabilities))
     (ros::ros-info (format nil "probabilities: ~A~%" plist))
     plist)))

(ros::roseus "move_for_verification")
(setq m (instance move-for-verification-cb-class :init))
(ros::spin)
