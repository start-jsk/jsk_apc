#!/usr/bin/env roseus
;; release-item.l
(load "package://baxtereus/baxter-interface.l")

(ros::roseus "move_arm_server")
(ros::load-ros-manifest "jsk_2015_05_baxter_apc")

(defun subscribe-init ()
  ;(unless (boundp '*tfl*) (setq *tfl* (instance ros::transform-listener :init)))
  (ros::advertise-service "/semi/release_item" jsk_2015_05_baxter_apc::ReleaseItem #'release-item-cb)
  (ros::ros-info "subscriber initialized")
  )

(defun release-item-cb (req)
  (ros::ros-info "release-item called")
  ; (send *ri* :head-pan -1.5) ;; see the target
  ; releasing bin upper pos
  (send *baxter* :angle-vector #f(19.7534 0.197754 -30.7397 1.97754 41.1328 -0.900879 72.5537 -0.241699 -23.1372 -71.5869 -32.2119 98.3716 14.6558 66.0498 -2.87842))
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (unix::sleep 3)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  ; releasing pos
  (send *baxter* :rarm :move-end-pos #f(0 0 -250) :world)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (unix::sleep 3)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  ; release item
  (send *ri* :stop-grasp)
  ; releasing bin upper pos
  (send *baxter* :rarm :move-end-pos #f(0 0 300) :world)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (unix::sleep 3)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  ; reset-pose
  (send *baxter* :reset-pose)
  ; (send *ri* :head-pan 0)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (unix::sleep 3)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  ; return response
  (setq res (send req :response))
  (send res :succeeded t)
  res
  )

(defun demo ()
  (baxter-init)
  ;(send *baxter* :locate #f(0 0 950) :world)
  (objects (list *baxter*))
  ;(baxter-reset-pose)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  ;(send *robot* :untuck-pose)
  ;(send *ri* :angle-vector (send *robot* :angle-vector))
  ;(unix::sleep 3)
  ;(setq *base* "/base")
  (subscribe-init)
  (ros::spin)
  )

(demo)