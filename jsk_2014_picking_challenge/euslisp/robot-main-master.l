#!/usr/bin/env roseus
;;
(ros::roseus "robot_main")

(ros::load-ros-manifest "jsk_recognition_msgs")
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
;; (load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
;; (load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(require :state-machine "package://roseus_smach/src/state-machine.l")
(require :state-machine-ros "package://roseus_smach/src/state-machine-ros.l")
(require :state-mcahine-utils "package://roseus_smach/src/state-machine-utils.l")
(require :state-machine-actionlib "package://roseus_smach/src/state-machine-actionlib.l")

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_2014_picking_challenge")


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
 :slots (target-arm opposite-arm target-list target-object action-client))

(defmethod robot-manipulation-state-machine
  (:init (arm target-list-given)
	 (setq target-arm arm)
	 (setq opposite-arm (if (equal arm :larm) :rarm :larm))
	 (setq target-list target-list-given)
	 (warn (format nil "Arm is ~A" arm))
	 (setq action-client (instance ros::simple-action-client :init
				(format nil "/picking_server/~A" (symbol-to-str arm)) jsk_2014_picking_challenge::EusCommandAction))
	 (send action-client :wait-for-server)
	 )
  (:apc-setup ()
	      t)
  (:check-target-list ()
		 (if (> (length target-list) 0)
		     t
		   :end-picking
		   ))
  (:init-pose ()
	      (send self :send-command-goal "init_pose" "")
	      )
  (:move-to-target ()
		   (setq target-object (pop target-list))
		   (warning-message 2 ";; Pick Out ~A in ~A ~%" target-object target-arm)
		   (send self :send-command-goal "move_to_target" target-object)
		   )
  (:pick-1-object ()
		  (send self :send-command-goal "pick_object" target-object)
		  )
  (:wait-place-object ()
		      t
		      )
  (:place-object ()
		 (send self :send-command-goal "place_object" target-object)
		 )
  (:pick-object-from-multi ()
			   (send self :send-command-goal "pick_object" target-object)
			   )
  (:wait-verification ()
		      t
		      )
  (:move-to-verification ()
			 (send self :send-command-goal "move_for_verification" target-object)
			 )
  (:return-object ()
		  (send self :send-command-goal "move_for_verification" target-object)
		  (send self :send-command-goal "return_object" target-object)
		  )
  (:send-command-goal (command target-bin)
		      (setq goal (instance jsk_2014_picking_challenge::EusCommandActionGoal :init))
		      (send goal :header :stamp (ros::time-now))
		      (send goal :goal_id :stamp (ros::time-now))
		      (send goal :goal :command command)
		      (send goal :goal :target_bin target-bin)
		      goal
		      ;; (send action-client :send-goal goal)
		      )
  (:get-action-client ()
		      action-client
		      )
  )

(defun make-parallel-state-machine ()
  ;; (setq *robot-arm-smach* (instance robot-manipulation-state-machine :init))
  (setq *robot-rarm-smach* (instance robot-manipulation-state-machine :init :rarm (list "a" "b")))
  (setq *robot-larm-smach* (instance robot-manipulation-state-machine :init :larm (list "g" "h")))
  (setq *sm*
        (make-state-machine
	 '((:apc-setup -> (:rarm-check-target-list :larm-check-target-list))

	   ;; (:rarm-check-target-list -> :rarm-init-pose)
	   (:rarm-init-pose -> :rarm-move-to-target )
	   (:rarm-move-to-target -> :rarm-pick-object-from-multi)
	   (:rarm-move-for-verification -> :rarm-wait-place-object)
	   (:rarm-wait-place-object -> :rarm-place-object)
	   (:rarm-return-object -> :rarm-pick-object-from-multi)
	   (:rarm-place-object -> :rarm-check-target-list)
	   (:rarm-pick-1-object -> :rarm-wait-place-object)
	   (:rarm-wait-verification -> :rarm-move-for-verification)

	   ;; (:larm-check-target-list -> :larm-init-pose)
	   (:larm-init-pose -> :larm-move-to-target )
	   (:larm-move-to-target -> :larm-pick-object-from-multi)
	   (:larm-move-for-verification -> :larm-wait-place-object)
	   (:larm-wait-place-object -> :larm-place-object)
	   (:larm-return-object -> :larm-pick-object-from-multi)
	   (:larm-place-object -> :larm-check-target-list)
	   (:larm-pick-1-object -> :larm-wait-place-object)
	   (:larm-wait-verification -> :larm-move-for-verification)
	   )
	 '((:apc-setup '(lambda (&rest args) (send *robot-rarm-smach* :apc-setup)))
	   ;; (:rarm-init-pose '(lambda (&rest args) (send *robot-rarm-smach* :init-pose)))
	   (:rarm-check-target-list '(lambda (&rest args) (send *robot-rarm-smach* :check-target-list)))
	   ;; (:rarm-move-to-target '(lambda (&rest args) (send *robot-rarm-smach* :move-to-target)))
	   ;; (:rarm-pick-object-from-multi '(lambda (&rest args) (send *robot-rarm-smach* :pick-object-from-multi)))
	   ;; (:rarm-move-for-verification '(lambda (&rest args) (send *robot-rarm-smach* :move-for-verification)))
	   ;; (:rarm-return-object '(lambda (&rest args) (send *robot-rarm-smach* :return-object)))
	   ;; (:rarm-place-object '(lambda (&rest args) (send *robot-rarm-smach* :place-object)))
	   ;; (:rarm-wait-verification '(lambda (&rest args) (send *robot-rarm-smach* :wait-verification)))
	   ;; (:rarm-wait-place-object '(lambda (&rest args) (send *robot-rarm-smach* :wait-place-object)))
	   ;; (:rarm-pick-1-object '(lambda (&rest args) (send *robot-rarm-smach* :pick-1-object)))


	   ;; (:larm-init-pose '(lambda (&rest args) (send *robot-larm-smach* :init-pose)))
	   (:larm-check-target-list '(lambda (&rest args) (send *robot-larm-smach* :check-target-list)))
	   ;; (:larm-move-to-target '(lambda (&rest args) (send *robot-larm-smach* :move-to-target)))
	   ;; (:larm-pick-object-from-multi '(lambda (&rest args) (send *robot-larm-smach* :pick-object-from-multi)))
	   ;; (:larm-move-for-verification '(lambda (&rest args) (send *robot-larm-smach* :move-for-verification)))
	   ;; (:larm-return-object '(lambda (&rest args) (send *robot-larm-smach* :return-object)))
	   ;; (:larm-place-object '(lambda (&rest args) (send *robot-larm-smach* :place-object)))
	   ;; (:larm-wait-verification '(lambda (&rest args) (send *robot-larm-smach* :wait-verification)))
	   ;; (:larm-wait-place-object '(lambda (&rest args) (send *robot-larm-smach* :wait-place-object)))
	   ;; (:larm-pick-1-object '(lambda (&rest args) (send *robot-larm-smach* :pick-1-object)))
	   )
	 '(:apc-setup)
	 '(:success)
	 :exec-result :succeeded
	 )
	)

  ;; NODES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  (send *sm* :add-node (actionlib-client-state
			:rarm-init-pose (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :init-pose))))
  ;; (send *sm* :add-node (actionlib-client-state
  ;; 		      :rarm-check-target-list (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
  ;; 		      :key '(lambda (&rest args) (send *robot-rarm-smach* :check-target-list))))
  (send *sm* :add-node (actionlib-client-state
			:rarm-move-to-target (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :move-to-target))))
  (send *sm* :add-node (actionlib-client-state
			:rarm-pick-object-from-multi (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :pick-object-from-multi))))
  (send *sm* :add-node (actionlib-client-state
			:rarm-move-for-verification (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :move-for-verification))))
  (send *sm* :add-node (actionlib-client-state
			:rarm-return-object (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :return-object))))
  (send *sm* :add-node (actionlib-client-state
			:rarm-place-object (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :place-object))))
  (send *sm* :add-node (actionlib-client-state
			:rarm-wait-verification (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :wait-verification))))
  (send *sm* :add-node (actionlib-client-state
			:rarm-wait-place-object (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :wait-place-object))))
  (send *sm* :add-node (actionlib-client-state
			:rarm-pick-1-object (send *robot-rarm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-rarm-smach* :pick-1-object))))


  (send *sm* :add-node (actionlib-client-state
			:larm-init-pose (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :init-pose))))
  ;; (send *sm* :add-node (actionlib-client-state
  ;; 		      :larm-check-target-list (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
  ;; 		      :key '(lambda (&rest args) (send *robot-larm-smach* :check-target-list))))
  (send *sm* :add-node (actionlib-client-state
			:larm-move-to-target (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :move-to-target))))
  (send *sm* :add-node (actionlib-client-state
			:larm-pick-object-from-multi (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :pick-object-from-multi))))
  (send *sm* :add-node (actionlib-client-state
			:larm-move-for-verification (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :move-for-verification))))
  (send *sm* :add-node (actionlib-client-state
			:larm-return-object (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :return-object))))
  (send *sm* :add-node (actionlib-client-state
			:larm-place-object (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :place-object))))
  (send *sm* :add-node (actionlib-client-state
			:larm-wait-verification (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :wait-verification))))
  (send *sm* :add-node (actionlib-client-state
			:larm-wait-place-object (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :wait-place-object))))
  (send *sm* :add-node (actionlib-client-state
			:larm-pick-1-object (send *robot-larm-smach* :get-action-client) :timeout 300 :retry t
			:key '(lambda (&rest args) (send *robot-larm-smach* :pick-1-object))))



  ;; NODES END ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


  ;; from and to nodes are selected by name or symbol
  (send *sm* :add-transition :rarm-check-target-list :rarm-init-pose t)
  (send *sm* :add-transition :larm-check-target-list :larm-init-pose t)


  (send *sm* :add-transition :rarm-pick-object-from-multi :rarm-pick-object-from-multi nil)
  (send *sm* :add-transition :larm-pick-object-from-multi :larm-pick-object-from-multi nil)

  (send *sm* :add-transition :rarm-move-for-verification :rarm-return-object nil)
  (send *sm* :add-transition :larm-move-for-verification :larm-return-object nil)

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



