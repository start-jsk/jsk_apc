#!/usr/bin/env roseus
;;;
;;; Move arm to target pick bin region in the shelf.
;;;
(load "package://baxtereus/baxter-interface.l")
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
    (let ((order (send goal :goal :order))
          (limb (send goal :goal :limb))
          (foldav-l #f(37.3096 -2.30713 -74.0698 146.667 99.5361 -3.66943 6.48193))
          (foldav-r #f(-48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745))
          avs target-nm seq c r msg)
      (setq target-nm (intern (string-upcase order)))
      (push "msg_validation" seq)
      ;; limb validation
      (cond ((string-equal limb "left") (setq limb :larm))
            ((string-equal limb "right") (setq limb :rarm))
            (t (progn
                 (push "invalid_limb" seq)
                 (setq msg (send server :result :sequence (nreverse seq)))
                 (send server :set-aborted msg)
                 (return-from :execute-cb nil))))
      (ros::ros-info (format nil "[move_arm2target_bin] requested ~A." target-nm))
      ;; target name validation
      (unless (find target-nm (send region :target-name-list))
        (ros::ros-info (format nil "[move_arm2target_bin] ~A not found." target-nm))
        (setq msg (send server :result :sequence (nreverse seq)))
        (send server :set-aborted msg)
        (return-from :execute-cb nil))
      ;; ik with z axis restriction
      (push (format nil "IK_for_~A" target-nm) seq)
      (ros::ros-info (format nil "[move_arm2target_bin] move arm to: ~A" target-nm))
      ;; original pose
      (push (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
      ;; to avoid shelf
      (send *baxter* :rarm :angle-vector foldav-r)
      (send *baxter* :larm :angle-vector foldav-l)
      (push (send *baxter* :angle-vector) avs)
      ;; go to in front of the target bin
      (setq c (send *pod* :transform-vector (send region :get-val target-nm)))
      (setq r (send *baxter* limb
                  :inverse-kinematics (make-cascoords :pos c)
                  :rotate-axis t :revert-if-fail nil))
      (if r (push r avs)
        (progn
          (if (equal limb :larm)
            (send *baxter* limb :angle-vector foldav-l)
            (send *baxter* limb :angle-vector foldav-r))
          (push (send *baxter* :angle-vector) avs)
          (push (send *baxter* limb :inverse-kinematics (make-cascoords :pos c)
                      :rotate-axis t :revert-if-fail nil) avs)))
      ;; reflect changes to *ri*
      (push "ri_reflection" seq)
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 8000))
      (send *ri* :wait-interpolation)
      (send *irtviewer* :draw-objects)
      ;; set result
      (setq msg (send server :result :sequence (nreverse seq)))
      (send server :set-succeeded msg)
      ))
  (:spin-once () (send self :worker) (send action :spin-once))
  (:spin () (do-until-key (send self :spin-once) (ros::sleep)))
  )

(ros::roseus "move_arm2target_bin")
(ros::rate 500)
(setq s (instance move2arm-actionlib-server :init))
(send s :spin)

