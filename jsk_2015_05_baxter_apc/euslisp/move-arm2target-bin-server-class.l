#!/usr/bin/env roseus
;;
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")

(ros::roseus "move_arm2target_bin")
(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defclass move2arm-actionlib-server
  :super propertied-object
  :slots (region (action :forward (:worker))))

(defmethod move2arm-actionlib-server
    (:init ()
     ;; baxter model
     (baxter-init)
     (send *baxter* :locate #f(0 0 950) :world)
     ;; pod-lowres model
     (when (not (boundp '*pod*))
       (load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
       (send *pod* :locate #f(1300 0 0) :world)
       (send *pod* :rotate pi/2 :x :world)
       (send *pod* :rotate (- pi/2) :z :world)
       )
     (objects (list *baxter* *pod*))
     (setq action (instance ros::simple-action-server :init
                            "/move_arm2target_bin" jsk_2014_picking_challenge::MoveArm2TargetBinAction
                            :execute-cb `(lambda (server goal) (send ,self :execute-cb server goal))))
     (setq region (instance target-pick-region :init)))
  (:execute-cb (server goal)
               """the order should be string and included in (a, b, c, ..., l)"""
               (let ((order (send goal :goal :order))
                     ;;(region (instance target-pick-region :init))
                     seq msg target-nm c target)
                 (push 0 seq)  ;; started
                 (setq target-nm (intern (string-upcase order)))
                 (ros::ros-info (format nil "moving to ~A." target-nm))
                 ;; target name validation
                 (push 1 seq)  ;; verifying target name
                 (unless (find target-nm (send region :target-name-list))
                   (ros::ros-info (format nil "~A not found." target-nm))
                   (return-from execute-cb nil))
                 ;; get target coordinates
                 (push 2 seq)  ;; getting target coordinates
                 (setq c (send *pod* :transform-vector (send region :get-val target-nm)))
                 ;; ik with z axis constraction
                 (push 3 seq)  ;; computing IK
                 (ros::ros-info (format nil "computing IK for ~A." target-nm))

                 (setq target (instance cascaded-coords :init :pos c))
                 (send *baxter* :larm :inverse-kinematics target :rotate-axis :z)
                 ;; reflect change to ri
                 (push 4 seq)  ;; reflecting to ri
                 (send *ri* :angle-vector (send *baxter* :angle-vector))
                 (send *irtviewer* :draw-objects)
                 (x::window-main-one)
                 ;; set result
                 (push msg (send server :result :sequence (nreverse seq)))
                 (send server :set-succeeded msg)
                 ))
  (:spin ()
         (ros::rate 500)
         (do-until-key
             (send self :worker)
           (send action :spin-once)
           (ros::sleep)))
  (:spin-once () (send self :worker) (send action :spin-once))
  )

(ros::rate 500)
(setq s (instance move2arm-actionlib-server :init))
(do-until-key
  (send s :spin))
