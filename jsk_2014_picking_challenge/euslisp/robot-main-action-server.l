#!/usr/bin/env roseus
;;
(ros::roseus "robot_main")

(ros::load-ros-manifest "jsk_recognition_msgs")
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

;;Global Variables
(setq *use-arm* nil)
(setq *arm-action-server* nil)
(setq *hogehoge* "inagaki")

;; end of Global Variables

(defun command-callback(server goal)
  (let* ((target-bin (send goal :goal :target_bin))
	 (target-bin-symbol (str-to-symbol target-bin))
	 (command    (send goal :goal :command))
	 (result (send server :result)))
    ;; get status from parameter server
    (send *irtviewer* :draw-objects)
    (ros::ros-info " command: ~A, arm: ~A, target-bin: ~A" command (arm-to-str *use-arm*) target-bin)
    (speak-en (format nil " command: ~A, arm: ~A, target-bin: ~A" command (symbol-to-str *use-arm*) target-bin) :google t)
    (cond
     ((string= command "init_pose")
      ;; (move-to-target-bin *use-arm* target-bin-symbol)
      (send server :set-succeeded result))
     ((string= command "return_object")
      ;; (move-to-target-bin *use-arm* target-bin-symbol)
      (send server :set-succeeded result))
     ((string= command "move_to_target")
      (move-to-target-bin *use-arm* target-bin-symbol)
      (send server :set-succeeded result))
     ((string= command "pick_object")
      (pick-object *use-arm*)
      (send server :set-succeeded result))
     ((string= command "move_for_verification")
      (move-for-verification *use-arm*)
      (send server :set-succeeded result))
     ((string= command "place_object")
      (place-object *use-arm*)
      (send server :set-succeeded result))
     (t (progn
	  (speak-en (format nil "Unknow command ~A had passed." command) :google t)
	  (warn (format nil "Unknow command ~A had passed.~%" command))
	  (send result :message (format nil "Unknow command ~A had passed." command))
	  (send server :set-aborted result)
	  )))))

(defun main ()
  (apc-init)
  (ros::rate 1)
  (while (ros::ok)
    (send *arm-action-server* :worker)
    (ros::spin-once)
    (ros::sleep)
    )
  )

; this is test workorder
; TODO: make workorder publisher
(setq workorder (list '("right" :c) '("left" :a) '("right" :b) '("left" :d) '("right" :e)))

(defun config-setup()
  (let ((arm (ros::get-param "~arm")))
    (warn "~% start config ~%")
    (setq *use-arm* (cond
		 ((equal arm "rarm") :rarm)
		 ((equal arm "larm") :larm) 
		 (t (progn
		      (warn "!! arm paramenters are not set correctly !! ~%")
		      (exit)
		      ))))
    (setq *arm-action-server*
	  (instance ros::simple-action-server :init (format nil "/picking_server/~A" arm)
		    jsk_2014_picking_challenge::EusCommandAction
		    :execute-cb `(lambda (s g) (command-callback s g))
		    ;; :preempt-cb `(lambda (s g) (send ,self :preempt-cb s g))
		    ;; :accept-cb  `(lambda (s g) (send ,self :accept-cb  s g))
		    ))
    )
  )

(config-setup)

(warn "~% Commands ~%")
(warn "(apc-init) : Setup~%")
(warn "(main)     : Start the loop~%")

