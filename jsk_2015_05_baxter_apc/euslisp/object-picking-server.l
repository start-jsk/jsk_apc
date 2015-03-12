#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")
(load "package://jsk_2014_picking_challenge/euslisp/gripper-control.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defclass object-picking-actionlib-server
  :super propertied-object
  :slots ((action :forward (:worker))))
(defmethod object-picking-actionlib-server
  (:init
    ()
    (baxter-init)
    (send *baxter* :locate #f(0 0 950) :world)
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (pod-init)
    (orderbin-init)
    (objects (list *baxter* *pod* *orderbin*))
    (setq action (instance ros::simple-action-server :init
                           "/object_picking" jsk_2014_picking_challenge::ObjectPickingAction
                           :execute-cb `(lambda (server goal) (send ,self :execute-cb server goal)))))
  (:execute-cb
    (server goal)
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (let ((l/r (send goal :goal :limb))
          (on/off (send goal :goal :state)) ;; if you want to return items, state must be "false"
          seq picking-coords msg)
      ;; l/r validataion
      (cond ((string-equal l/r "left") (setq l/r ':larm))
            ((string-equal l/r "right") (setq l/r ':rarm))
            (t (progn
                 (push "invalid_l/r" seq)
                 (setq msg (send server :result :sequence (nreverse seq)))
                 (send server :set-aborted msg)
                 (return-from :execute-cb nil))))
      ;; picking pose
      (push "goint to pick" seq)
      (ros::ros-info (format nil "[object_picking] going to pick object"))
      (setq avs nil)
      (push (send *baxter* :angle-vector) avs)
      (push (send *baxter* l/r :move-end-pos #f(0 0 50) :world :revert-if-fail nil) avs)
      (push (send *baxter* l/r :move-end-pos #f(150 0 0) :world :revert-if-fail nil) avs)
      (push (send *baxter* l/r :move-end-pos #f(0 0 -50) :world :revert-if-fail nil) avs)
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 5000)) 
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; go/stop grasp
      (push "go/stop grasp" seq)
      (ros::ros-info (format nil "[object_picking] cleaner ~A" (if on/off "ON" "OFF")))
      (send *ri* (if on/off :go-grasp :stop-grasp))
      ;; return to original pose
      (push "returning to original pose" seq)
      (ros::ros-info (format nil "[object_picking] returning to original pose."))
      (setq avs (reverse avs))
      (pop avs)
      (pop avs)
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 5000))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; set result
      (ros::ros-info (format nil "~A" seq))
      (setq msg (send server :result :sequence (nreverse seq)))
      (send server :set-succeeded msg)
      ))
  (:spin-once () (send self :worker) (send action :spin-once))
  (:spin () (do-until-key (send self :spin-once) (ros::sleep))))

(ros::roseus "object_picking")
(ros::rate 500)
(setq s (instance object-picking-actionlib-server :init))
(send s :spin)

