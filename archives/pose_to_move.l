#!/usr/bin/env roseus

(ros::roseus "move_arm_server")
(ros::load-ros-manifest "jsk_2015_05_baxter_apc")

(defun baxter-ik-init ()
  (load "package://baxtereus/baxter-interface.l")
  (baxter-init)
  ;(send *baxter* :locate #f(0 0 950) :world)
  (objects (list *baxter*))
  (baxter-reset-pose)
  (setq *robot* *baxter*)
  (setq *base* "/base")
  (subscribe-init)
  )

(defun baxter-reset-pose ()
  (send *baxter* :larm :shoulder-y :joint-angle -30)
  (send *baxter* :larm :shoulder-p :joint-angle -30)
  (send *baxter* :larm :elbow-r  :joint-angle 30)
  (send *baxter* :larm :elbow-p  :joint-angle 90)
  (send *baxter* :larm :wrist-r  :joint-angle 30)
  (send *baxter* :larm :wrist-p  :joint-angle -30)
  (send *baxter* :head :neck-y   :joint-angle -30)
  (objects (list *baxter*))
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  )


(defun pr2-ik-init ()
  (load "package://pr2eus/pr2-interface.l")
  (pr2-init)
  (send *pr2* :angle-vector
         (float-vector 12.0 45.0 60.0 0.0 -120.0 0.0 -45.0 0.0 -45.0 60.0 0.0 -120.0 0.0 -45.0 0.0 0.0 35.0))

  
  (send *ri* :angle-vector (send *pr2* :angle-vector))
  (setq *robot* *pr2*)
  (setq *base* "/base_link")
  (subscribe-init)
  )
(defun subscribe-init ()
  (unless (boundp '*tfl*)
    (setq *tfl* (instance ros::transform-listener :init)));;;_??????
  (ros::subscribe "/move_left_arm" geometry_msgs::PoseStamped #'start_move_left) 
  (ros::subscribe "/move_right_arm" geometry_msgs::PoseStamped #'start_move_right) 
  (ros::advertise-service "/move_left_arm_service" jsk_2015_05_baxter_apc::MoveArm #'service-right-cb)
  (ros::advertise-service "/move_right_arm_service" jsk_2015_05_baxter_apc::MoveArm #'service-left-cb)
  (ros::ros-info "subscriber initialized")
  (ros::spin)   ;;;;?????????
  )

;;(baxter-init)
;;(pr2-init)


(defun start_move_left (msg)
  (ros::ros-info "start_grasp_cb driven")
  (start_move msg :larm)
  )
(defun start_move_right (msg)
  (ros::ros-info "start_grasp_cb driven")
  (start_move msg :rarm)
)
(defun service-left-cb (req)
  (let ((tgt (send req :pose_arm)))
    ;; move left arm 
    (setq res (send req :response))
    (send res :succeeded (start_move tgt :larm))
    res
  ))
(defun service-right-cb (req)
  (let ((tgt (send req :pose_arm)))
    ;; move left arm 
    (setq res (send req :response))
    (send res :succeeded (start_move tgt :rarm))
    res
  ))


(defun start_move (msg arm)  ;;;;;;????????????
  (setq trans (send *tfl* :lookup-transform
		    *base*
		    (send msg :header :frame_id)
		    (ros::time 0.0);;(send *msg* :header :stamp)
		    ))
  (setq cds (send (send trans :copy-worldcoords) :transform (ros::tf-pose->coords(send msg :pose)))) 
  (setq ret (send *robot* arm :inverse-kinematics cds))
  (if ret
      (progn 
	(send *ri* :angle-vector (send *robot* :angle-vector))
	(send *ri* :wait-interpolation)
  ;(objects cds)  
	)
  (progn (ros::ros-info "failer"))
  )
  ret
  )

(warn 
"
;pr2-ik-init
;baxter-ik-init
"
)


