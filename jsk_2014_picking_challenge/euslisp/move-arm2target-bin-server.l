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
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (let ((order (send goal :goal :order))
          (limb (send goal :goal :limb))
          (avs nil) target-nm seq c msg untuck-coords)
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
      (ros::ros-info (format nil "requested ~A." target-nm))
      ;; target name validation
      (unless (find target-nm (send region :target-name-list))
        (ros::ros-info (format nil "~A not found." target-nm))
        (setq msg (send server :result :sequence (nreverse seq)))
        (send server :set-aborted msg)
        (return-from :execute-cb nil))
      ;; ik with z axis restriction
      (push (format nil "inverse_kinematics_for_~A" target-nm) seq)
      (ros::ros-info (format nil "computing IK for ~A." target-nm))
      (push (send *baxter* :angle-vector) avs)  ;; set original av
      (setq untuck-coords (make-cascoords :pos (float-vector 550 (if (equal limb :larm) 200 -200) 1100)
                                          :rot #2f((-3.6e-06 0 -1) (0 1 0) (1 0 -3.6e-06))))
      (push (send *baxter* limb :inverse-kinematics untuck-coords
                  :rotation-axis :z :revert-if-fail nil) avs)
      ;; get target coordinates
      (push "get_target_coords" seq)
      (setq c (send *pod* :transform-vector (send region :get-val target-nm)))
      (push (send *baxter* limb
                  :inverse-kinematics (make-cascoords :pos c)
                  :rotate-axis t :revert-if-fail nil) avs)
      (warn (format nil "~A~%" avs))
      ;; reflect change to ri
      (push "ri_reflectioin" seq)  ;; reflecting to ri
      (send *ri* :angle-vector-sequence (reverse avs) (make-list (length avs) :initial-element 10000))
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
