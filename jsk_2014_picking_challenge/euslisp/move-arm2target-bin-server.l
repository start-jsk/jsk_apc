#!/usr/bin/env roseus
;;;
;;; Move arm to target pick bin region in the shelf.
;;;
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/arm-motion.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(defclass move2arm-actionlib-server
  :super propertied-object
  :slots (region (action :forward (:worker))))
(defmethod move2arm-actionlib-server
  (:init
    ()
    ;; baxter model
    (baxter-init)
    (send *baxter* :locate #f(0 0 950) :world)
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (pod-init)
    (orderbin-init)
    (objects (list *baxter* *pod* *orderbin*))
    (setq action (instance ros::simple-action-server
                           :init "/move_arm2target_bin" jsk_2014_picking_challenge::MoveArm2TargetBinAction
                           ;; args for callback function are needed
                           ;; to avoid segmentation fault
                           :execute-cb `(lambda (server goal) (send ,self :execute-cb server goal))))
    (setq region (instance target-pick-region :init)))
  (:execute-cb
    (server goal)
    """the order should be string and included in (a, b, c, ..., l)"""
    (let* ((arm (str2arm (send goal :goal :limb)))
           (bin (str2symbol (send goal :goal :order))) avs seq)
      ;; validate limb
      (ros::ros-info "validate limb")
      (unless (validate-arm arm)
        (send server :set-aborted (send server :result :sequence seq))
        (return-from :execute-cb nil))
      (ros::ros-info "validate bin")
      ;; validate target bin name
      (unless (validate-bin-name bin)
        (send server :set-aborted (send server :result :sequence seq))
        (return-from :execute-cb nil))
      (ros::ros-info "move arm to target bin")
      (move-arm-to-target-bin arm bin)
      (send *irtviewer* :draw-objects)
      ;; set result
      (send server :set-succeeded (send server :result :sequence seq))
      ))
  ;(:spin-once () (send self :worker) (send action :spin-once))
  ;(:spin () (do-until-key (send self :spin-once) (ros::sleep)))
  )

(ros::roseus "move_arm2target_bin")
(ros::rate 500)
(setq s (instance move2arm-actionlib-server :init))
;(send s :spin)
(setq r (ros::rate 10))
(do-until-key (ros::ok)
              (send *ri* :spin-once) ;; (ros::spin-once "baxter_interface")
              (send s :worker)
              (ros::spin-once)
              (ros::sleep))

