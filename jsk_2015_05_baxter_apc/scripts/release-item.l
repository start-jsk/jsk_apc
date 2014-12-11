#!/usr/bin/env roseus
;; release-item.l
(load "package://baxtereus/baxter-interface.l")

(ros::roseus "move_arm_server")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defun subscribe-init ()
  ;(unless (boundp '*tfl*) (setq *tfl* (instance ros::transform-listener :init)))
  (ros::advertise-service "/semi/release_item" jsk_2014_picking_challenge::ReleaseItem #'release-item-cb)
  (ros::ros-info "subscriber initialized")
  )

(defun release-item-cb (req)
  (ros::ros-info "release-item called")
  ; releasing bin upper pos
  (send *baxter* :angle-vector #f(-5.31738 -0.307617 -31.7725 -0.32959 42.8247 -0.241699 71.8506 -0.197754 -44.8901 -69.0161 -9.55811 107.007 7.91016 48.5376 -10.3931))
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (unix::sleep 3)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  ; releasing pos
  (send *baxter* :rarm :move-end-pos #f(0 0 -300) :world)
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