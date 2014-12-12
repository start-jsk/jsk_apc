#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::roseus "qrcode_pos_server")
(ros::load-ros-manifest "jsk_2014_picking_challenge")
(ros::load-ros-manifest "zbar_ros")

(defclass baxter-qrcode-pos-server
    :super propertied-object
  :slots (current-detected-marker baxter-angles))

(defmethod baxter-qrcode-pos-server
    (:init ()
     (ros::ros-info "baxter's qrcode-pos-server start initialization")
     (load "package://baxtereus/baxter-interface.l")
     (baxter-init)
     (objects (list *baxter*))
     (send *baxter* :untuck-pose)
     (send *ri* :angle-vector (send *baxter* :angle-vector))

     (ros::ros-info "baxter's marker-subscriber initialized")

     ;; /markers subscribe initalization
     (setq current-detected-marker "bin_E")
     (ros::subscribe "marker" zbar_ros::Marker
                     #'(lambda (msg)
                         (ros::ros-info (format nil "Marker [~A] detected~%" (send msg :data)))
                         (setq current-detected-marker (send msg :data))
                         )
                     )

     ;; service initialization;
     (ros::advertise-service "/semi/qrcode_pos"
                             jsk_2014_picking_challenge::QrStampsrv
                             #'send self :baxter-qrcode-pos-callback)
     (ros::ros-info "baxter's qrcode-pos-server initialized")
     )

  (:baxter-qrcode-pos-callback (request)
                               (let
                                   ((baxter-angles (list
                                                    #f(-7.44873 -5.03174 -56.9751 -68.0273 110.984 37.9907 59.0405 -0.021973 21.731 -37.3535 47.6367 74.5752 -60.2271 43.5498 39.0015)
                                                         )))
                                 (setq return-qrstampes
                                       (instance jsk_2014_picking_challenge::QrStampes :init))
                                 (dolist (angle baxter-angles)
                                   (setq stamp
                                         (instance jsk_2014_picking_challenge::QrStamp :init))
                                   (format t "~A" angle)

                                   (send *baxter* :angle-vector angle)
                                   (send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
                                   (unix:sleep 3)
                                   (send *irtviewer* :draw-objects)

                                   (setq tmp (ros::coords->tf-pose-stamped
                                              (send *baxter* :rarm :end-coords) "base"))
                                   (setq m (instance jsk_2014_picking_challenge::QrStamp
                                                     :init
                                                     :label (instance std_msgs::String :init :data current-detected-marker)
                                                     :qrcode_pose_stamp tmp))
                                   (send return-qrstampes :qrcode_stampes
                                         (append (send return-qrstampes :qrcode_stampes)
                                                 (list m)))
                                   (unix:sleep 5)
                                   )
                                 return-qrstampes)
                               )
  )

(setq b (instance baxter-qrcode-pos-server :init))
(do-until-key
    (ros::spin-once))
