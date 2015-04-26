#!/usr/bin/env roseus
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
;; (load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
;; (load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(require :state-machine "package://roseus_smach/src/state-machine.l")
(require :state-machine-ros "package://roseus_smach/src/state-machine-ros.l")
(require :state-mcahine-utils "package://roseus_smach/src/state-machine-utils.l")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(ros::roseus "robot_main")

;;Global Variables

;; end of Global Variables

;; (setq client (instance ros::simple-action-client :init
;; 		  "/picking_server/rarm" jsk_2014_picking_challenge::EusCommandAction))
;; (ros::ros-info "waiting ~A available" "picking_server/rarm")
;; (send client :wait-for-server)

;; (setq goal (instance jsk_2014_picking_challenge::EusCommandActionGoal :init))
;; (send goal :header :stamp (ros::time-now))
;; (send goal :goal_id :stamp (ros::time-now))
;; (send goal :goal :command  "move_to_target")
;; (send goal :goal :target_bin "c")
;; (send client :send-goal goal))

;; (send client :wait-for-result)

(defclass robot-manipulation-state-machine
 :super propertied-object
 :slots (target-arm opposite-arm target-list target-object))

(defmethod robot-manipulation-state-machine
  (:init (arm target-list-given)
	 (setq target-arm arm)
	 (setq opposite-arm (if (equal arm :larm) :rarm :larm))
	 (setq target-list target-list-given)
	 )
  (:apc-setup ()
   t)
  (:check-target-list ()
		 (if (> (length target-list) 0)
		     t
		   :end-picking
		   ))
  (:init-pose ()
	      t
	      )
  (:move-to-target ()
		   (setq target-object (pop target-list))
		   (warning-message 2 ";; Pick Out ~A in ~A ~%" target-object target-arm)
		   t
		   )
  (:pick-1-object ()
		  t
		  )
  (:wait-place-object ()
		      t
		      )
  (:place-object ()
		 t
		 )
  (:pick-object-from-multi ()
			   t
			   )
  (:wait-verification ()
		      t
		      )
  (:move-to-verification ()
			 t)
  (:return-object ()
		  t)
  )

(defun make-parallel-state-machine ()
  ;; (setq *robot-arm-smach* (instance robot-manipulation-state-machine :init))
  (setq *robot-rarm-smach* (instance robot-manipulation-state-machine :init :rarm (list "a" "b")))
  (setq *robot-larm-smach* (instance robot-manipulation-state-machine :init :larm (list "g" "h")))
  (setq *sm*
        (make-state-machine
	 '((:apc-setup -> (:rarm-check-target-list :larm-check-target-list))
	   (:rarm-check-target-list -> :rarm-init-pose)
	   (:rarm-init-pose -> :rarm-move-to-target )
	   (:rarm-move-to-target -> :rarm-pick-object-from-multi)
	   (:rarm-move-to-verification -> :rarm-wait-place-object)
	   (:rarm-wait-place-object -> :rarm-place-object)
	   (:rarm-return-object -> :rarm-pick-object-from-multi)
	   (:rarm-place-object -> :rarm-check-target-list)
	   (:rarm-pick-1-object -> :rarm-wait-place-object)
	   (:rarm-wait-verification -> :rarm-move-to-verification)

	   (:larm-check-target-list -> :larm-init-pose)
	   (:larm-init-pose -> :larm-move-to-target )
	   (:larm-move-to-target -> :larm-pick-object-from-multi)
	   (:larm-move-to-verification -> :larm-wait-place-object)
	   (:larm-wait-place-object -> :larm-place-object)
	   (:larm-return-object -> :larm-pick-object-from-multi)
	   (:larm-place-object -> :larm-check-target-list)
	   (:larm-pick-1-object -> :larm-wait-place-object)
	   (:larm-wait-verification -> :larm-move-to-verification)
	   )
	 '((:apc-setup '(lambda (&rest args) (send *robot-rarm-smach* :apc-setup)))
	   (:rarm-init-pose '(lambda (&rest args) (send *robot-rarm-smach* :init-pose)))
	   (:rarm-check-target-list '(lambda (&rest args) (send *robot-rarm-smach* :check-target-list)))
	   (:rarm-move-to-target '(lambda (&rest args) (send *robot-rarm-smach* :move-to-target)))
	   (:rarm-pick-object-from-multi '(lambda (&rest args) (send *robot-rarm-smach* :pick-object-from-multi)))
	   (:rarm-move-to-verification '(lambda (&rest args) (send *robot-rarm-smach* :move-to-verification)))
	   (:rarm-return-object '(lambda (&rest args) (send *robot-rarm-smach* :return-object)))
	   (:rarm-place-object '(lambda (&rest args) (send *robot-rarm-smach* :place-object)))

	   (:rarm-wait-verification '(lambda (&rest args) (send *robot-rarm-smach* :wait-verification)))
	   (:rarm-wait-place-object '(lambda (&rest args) (send *robot-rarm-smach* :wait-place-object)))
	   (:rarm-pick-1-object '(lambda (&rest args) (send *robot-rarm-smach* :pick-1-object)))


	   (:larm-check-target-list '(lambda (&rest args) (send *robot-larm-smach* :check-target-list)))
	   (:larm-init-pose '(lambda (&rest args) (send *robot-larm-smach* :init-pose)))
	   (:larm-move-to-target '(lambda (&rest args) (send *robot-larm-smach* :move-to-target)))
	   (:larm-pick-object-from-multi '(lambda (&rest args) (send *robot-larm-smach* :pick-object-from-multi)))
	   (:larm-move-to-verification '(lambda (&rest args) (send *robot-larm-smach* :move-to-verification)))
	   (:larm-return-object '(lambda (&rest args) (send *robot-larm-smach* :return-object)))
	   (:larm-place-object '(lambda (&rest args) (send *robot-larm-smach* :place-object)))

	   (:larm-wait-verification '(lambda (&rest args) (send *robot-larm-smach* :wait-verification)))
	   (:larm-wait-place-object '(lambda (&rest args) (send *robot-larm-smach* :wait-place-object)))

	   (:larm-pick-1-object '(lambda (&rest args) (send *robot-larm-smach* :pick-1-object)))
	   )
	 '(:apc-setup)
	 '(:success)
	 )
	)
  ;; from and to nodes are selected by name or symbol
  (send *sm* :add-transition :rarm-pick-object-from-multi :rarm-pick-object-from-multi nil)
  (send *sm* :add-transition :larm-pick-object-from-multi :larm-pick-object-from-multi nil)

  (send *sm* :add-transition :rarm-move-to-verification :rarm-return-object nil)
  (send *sm* :add-transition :larm-move-to-verification :larm-return-object nil)

  (send *sm* :add-transition :rarm-pick-object-from-multi :rarm-wait-verification t)
  (send *sm* :add-transition :larm-pick-object-from-multi :larm-wait-verification t)

  (send *sm* :add-transition :rarm-wait-verification :rarm-wait-verification :wait)
  (send *sm* :add-transition :larm-wait-verification :larm-wait-verification :wait)

  (send *sm* :add-transition :rarm-wait-place-object :rarm-wait-place-object nil)
  (send *sm* :add-transition :larm-wait-place-object :larm-wait-place-object nil)
  
  (send *sm* :add-transition :rarm-check-target-list :success :end-picking)
  (send *sm* :add-transition :larm-check-target-list :success :end-picking)
  
  (send *sm* :add-transition :rarm-move-to-target :rarm-pick-1-object nil)
  (send *sm* :add-transition :larm-move-to-target :larm-pick-1-object nil)
  )

(defun init ()
  (ros::roseus "arm_state_machine_master")
  (make-parallel-state-machine)
  (ros::ros-info "created state machine ~A" *sm*))

(warning-message 3 ";; (init)~%")

(defun demo ()
  (when (or (not (boundp '*sm*)) (not (boundp '*robot-rarm-smach*) ) (not (boundp '*robot-larm-smach*)))
    (init))
  (exec-smach-with-spin *sm*))

(warning-message 3 ";; (demo)~%")

(provide :paralell-state-machine)
