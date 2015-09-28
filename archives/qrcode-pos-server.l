#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::roseus "qrcode_pos_server")
(ros::load-ros-manifest "jsk_2015_05_baxter_apc")
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
                             jsk_2015_05_baxter_apc::QrStampsrv
                             #'send self :baxter-qrcode-pos-callback)
     (ros::ros-info "baxter's qrcode-pos-server initialized")
     )

  (:baxter-qrcode-pos-callback (request)
                               (let
                                   ((baxter-angles (list
                                                     #f(-7.55859 20.061 -25.0049 -40.0342 60.0073 -19.9731 79.541 0.065918 75.498 -33.772 -54.9097 94.4824 -15.8643 -39.0234 57.8979)
                                                     #f(-7.58057 20.061 -25.0269 -40.0342 59.9854 -19.9731 79.519 0.021973 72.8394 -39.0674 -20.3906 90.7251 25.4883 -47.395 10.1953)
                                                         )))
                                 (setq return-qrstampes
                                       (instance jsk_2015_05_baxter_apc::QrStampes :init))
                                 (setq pm -1)
                                 (dolist (angle baxter-angles)
                                   (setq stamp
                                         (instance jsk_2015_05_baxter_apc::QrStamp :init))
                                   (format t "~A" angle)

                                   (send *baxter* :angle-vector angle)
                                   ; (send *ri* :head-pan (* pm 0.3))
                                   (send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
                                   (unix:sleep 3)
                                   (send *irtviewer* :draw-objects)

                                   (setq tmp (ros::coords->tf-pose-stamped
                                              (send *baxter* :rarm :end-coords) "base"))
                                   (setq m (instance jsk_2015_05_baxter_apc::QrStamp
                                                     :init
                                                     :label (instance std_msgs::String :init :data current-detected-marker)
                                                     :qrcode_pose_stamp tmp))
                                   (send return-qrstampes :qrcode_stampes
                                         (append (send return-qrstampes :qrcode_stampes)
                                                 (list m)))
                                   (unix:sleep 5)
                                   (setq pm (* pm -1))
                                   )
                                 ; (send *ri* :head-pan 0)
                                 return-qrstampes)
                               )
  )

(setq b (instance baxter-qrcode-pos-server :init))
(do-until-key
    (ros::spin-once))
