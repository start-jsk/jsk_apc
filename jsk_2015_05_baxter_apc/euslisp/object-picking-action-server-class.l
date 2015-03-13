#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/gripper-control.l")

(ros::roseus "object_picking")
(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defclass object-picking-actionlib-server
  :super propertied-object
  :slots ((action :forward (:worker))))

(defmethod object-picking-actionlib-server
    (:init ()
     (baxter-init)
     (send *baxter* :locate #f(0 0 950) :world)
     (when (not (boundp '*pod*))
       (load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
       (send *pod* :locate #f(1300 0 0) :world)
       (send *pod* :rotate pi/2 :x :world)
       (send *pod* :rotate -pi/2 :z :world))
     (objects (list *baxter* *pod*))
     (setq action (instance ros::simple-action-server :init
                            "/object_picking" jsk_2014_picking_challenge::ObjectPickingAction
                            :execute-cb `(lambda (server goal) (send ,self :execute-cb server goal)))))
  (:execute-cb (server goal)
   (let ((l/r (send goal :goal :limb))
         (on/off (send goal :goal :state)) ;; if you want to return items, state must be "false"
         seq endcoords msg)
     ;; l/r validataion
     (cond ((string-equal l/r "larm") (setq l/r ':larm))
           ((string-equal l/r "rarm") (setq l/r ':rarm))
           (t (progn
                (push "invalid_l/r" seq)
                (setq msg (send server :result :sequence (nreverse seq)))
                (send server :set-aborted msg)
                (return-from :execute-cb nil))))

     ;; get current arm's end-coords
     (push "get end-coords" seq)
     (setq endcoords
           (make-cascoords :pos (send *baxter* l/r :end-coords :worldpos)))
     (ros::ros-info (format nil "get ~A's endcoords." l/r))
     (ros::ros-info (format nil "~A" endcoords))

     ;; set target end-coords position
     (push "set target end-coords" seq)
     (setq endcoords
           (send endcoords :locate (float-vector (random 300) (random 50) (random 50))))
     (ros::ros-info (format nil "endcoords set."))
     (ros::ros-info (format nil "~A" endcoords))

     ;; ik with z axis constraction
     (push "ik computing" seq)
     (ros::ros-info (format nil "computing ik."))
     (send *baxter* l/r :inverse-kinematics endcoords :rotation-axis :z)

     ;; cleaner on/off
     (push "cleaner on/off" seq)
     (ros::ros-info (format nil "cleaner ~A" (if on/off "ON" "OFF")))
     (send *ri* (if on/off :start-grasp :stop-grasp))

     ;; apply changes to *ri*
     (push "apply changes" seq)
     (send *ri* :angle-vector (send *baxter* :angle-vector))
     (send *ri* :wait-interpolation)
     (send *irtviewer* :draw-objects)
     (x::window-main-one)

     ;; set result
     (ros::ros-info (format nil "~A" seq))
     (setq msg (send server :result :sequence (nreverse seq)))
     (send server :set-succeeded msg)
         ))
  (:spin-once () (send self :worker) (send action :spin-once))
  (:spin () (do-until-key (send self :spin-once) (ros::sleep))))

(ros::rate 500)
(setq s (instance object-picking-actionlib-server :init))
(send s :spin)
