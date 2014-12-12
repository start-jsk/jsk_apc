#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::roseus "qrcode-pos-server")
(ros::load-ros-manifest "jsk_2014_picking_challenge")
(ros::load-ros-manifest "")

(defclass baxter-qrcode-pos-server
    :super propertied-object
  :slots (current-detected-marker baxter-angles))

(defmethod baxter-qrcode-pos-server
    (:init ()
     (load "package://baxtereus/baxter-interface.l")
     (baxter-init)
     (objects (list *baxter*))
     (send *baxter* :untuck-pose)
     (send *ri* :angle-vector (send *baxter* :angle-vector))

     ;; /markers subscribe initalization
     (setq current-detected-marker "bin_E")
     (ros::subscribe "marker" zbar_ros::Marker
                     #'(lambda (msg)
                         (setq current-detected-marker (send msg :data))
                         (ros::ros-info (format nil "Marker [~A] detected~%" (send msg :data))))
                     )

     ;; service initialization
     (ros::advertise-service "/semi/qrcode_pos"
                             jsk_2014_picking_challenge::Qrstampsrv
                             #'send self :baxter-qrcode-pos-callback)
     (ros::ros-info "baxter's qrcode-pos-server initialized")
     (ros::spin)
     )

  (:move-arm (angle &key (move-duration-time 4000))
             (send *baxter* :angle-vector angle)
             (send *ri* :angle-vector (send *baxter* :angle-vector) move-duration-time)
             (unix:sleep 3)
             (send *irtviewer* :draw-objects)
             )

  (:baxter-qrcode-pos-callback (request)
                               (let (return-qrstampes)
                                 (setq return-qrstampes
                                       (instance jsk_2014_picking_challenge::QrStampes :init))
                                 (dolist (angle baxter-angles)
                                   (setq stamp
                                         (instance jsk_2014_picking_challenge::QrStamp :init))
                                   (move-arm angle)
                                   (setq tmp (ros::coords->tf-pose-stamped
                                              (send *baxter* :rarm :end-coords)))
                                   (send return-qrstampes :qrcode_stampes
                                         (cons stamp (send return-qrstampes :qrcode_stampes)))
                                   (send return-qrstampes :qrcode_stampes
                                         (append (send return-qrstampes :qrcode_stampes)
                                                 (list tmp)))
                                   )
                                 return-qrstampes)
                               )
  )
