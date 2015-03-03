#!/usr/bin/env roseus
;;
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")

(ros::roseus "object_picking")
(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defclass object-picking-actionlib-server
  :super propertied-object
  :slots (region (action :forward (:worker))))

(defmethod object-picking-actionlib-server
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
                            "/object_picking" jsk_2014_picking_challenge::ObjectPickingAction
                            :execute-cb `(lambda (server goal) (send ,self :execute-cb server goal))))
     (setq region (instance target-pick-region :init)))
  (:execute-cb (server goal)
   (let ((limb (send goal :goal :limb)) target-nm seq c msg)
     (push "msg_validation" seq)
     ;; limb validation
     (cond ((string-equal limb "left") (setq limb ':larm))
           ((string-equal limb "right") (setq limb ':rarm))
           (t (progn
                (push "invalid_limb" seq)
                (setq msg (send server :result :sequence (nreverse seq)))
                (send server :set-aborted msg)
                (return-from :execute-cb nil))))
     ;; (let ((order (send goal :goal :order))
     ;;       ;;(region (instance target-pick-region :init))
     ;;       seq msg target-nm c target)
     (push 0 seq)  ;; started
     (setq c (send *baxter* :rarm :end-coords))
     (ros::ros-info (format nil "get arm's endcoords."))

     ;; set end-coords position
     (push 1 seq)
     (setq c (send c :locate #f(30 0 50)))
     ;;(setq target (instance cascaded-coords :init :pos c))
     (ros::ros-info (format nil "endcoords set."))

     ;; ik with z axis constraction
     (push 2 seq)  ;; computing IK
     (ros::ros-info (format nil "computing IK."))
     (send *baxter* limb :inverse-kinematics c :rotate-axis :z)

     ;; reflect change to ri
     (push 3 seq)  ;; reflecting to ri
     (send *ri* :angle-vector (send *baxter* :angle-vector))
     (send *ri* :wait-interpolation)
     (send *irtviewer* :draw-objects)
     (x::window-main-one)

     ;; set result
     (setq msg (send server :result :sequence (nreverse seq)))
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
(setq s (instance object-picking-actionlib-server :init))
(do-until-key
  (send s :spin))
