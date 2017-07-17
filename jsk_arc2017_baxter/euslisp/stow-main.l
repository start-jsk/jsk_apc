#!/usr/bin/env roseus

(ros::roseus "robot_main")

(require "package://jsk_arc2017_baxter/euslisp/lib/stow-interface.l")
;; smach
(require :state-machine "package://roseus_smach/src/state-machine.l")
(require :state-machine-ros "package://roseus_smach/src/state-machine-ros.l")
(require :state-machine-utils "package://roseus_smach/src/state-machine-utils.l")

(defun make-stowing-state-machine ()
  (setq *sm*
        (make-state-machine
          '((:init -> :recognize-bboxes)
            (:recognize-bboxes -> :wait-for-user-input)
            (:wait-for-user-input -> :wait-for-opposite-arm)
            (:wait-for-user-input !-> :finish)
            (:wait-for-opposite-arm -> :recognize-object)
            (:wait-for-opposite-arm !-> :wait-for-opposite-arm)
            (:recognize-object -> :set-grasp-style)
            (:recognize-object !-> :return-from-recognize-object)
            (:return-from-recognize-object -> :check-recognize-fail-count)
            (:check-recognize-fail-count -> :wait-for-user-input)
            (:check-recognize-fail-count !-> :wait-for-opposite-arm-start-picking)
            (:set-grasp-style -> :pick-object)
            (:pick-object -> :verify-object)
            (:pick-object !-> :check-picking-fail-count)
            (:check-picking-fail-count -> :return-from-pick-object)
            (:check-picking-fail-count !-> :set-grasp-style)
            (:return-from-pick-object -> :wait-for-opposite-arm-start-picking)
            (:verify-object -> :set-target-bin)
            (:verify-object !-> :return-object)
            (:return-object -> :wait-for-opposite-arm)
            (:set-target-bin -> :place-object)
            (:set-target-bin !-> :return-object)
            (:place-object -> :return-from-place-object)
            (:return-from-place-object -> :wait-for-opposite-arm)
            (:wait-for-opposite-arm-start-picking -> :wait-for-opposite-arm)
            (:wait-for-opposite-arm-start-picking !-> :wait-for-opposite-arm-start-picking))
          '((:init
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :init)
                 (ros::ros-info "start ~a picking" (arm2str *arm*))
                 t))
            (:recognize-bboxes
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :recognize-bboxes)
                 (send *ti* :recognize-bboxes *arm*)
                 t))
            (:wait-for-user-input
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :wait-for-user-input)
                 (send *ti* :wait-for-user-input *arm*)))
            (:wait-for-opposite-arm
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :wait-for-opposite-arm)
                 (send *ti* :check-can-start *arm* :recognize-object
                       :wait-for-opposite-arm)))
            (:recognize-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :recognize-object)
                 (send *ti* :recognize-object *arm* :trial-times 10)))
            (:return-from-recognize-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :return-from-recognize-object)
                 (send *ti* :return-from-recognize-object *arm*)
                 t))
            (:check-recognize-fail-count
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-recognize-fail-count)
                 (send *ti* :check-recognize-fail-count *arm*)))
            (:set-grasp-style
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :set-grasp-style)
                 (send *ti* :set-grasp-style *arm*)
                 t))
            (:pick-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :pick-object)
                 (send *ti* :pick-object *arm*)))
            (:return-from-pick-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :return-from-pick-object)
                 (send *ti* :return-from-pick-object *arm*)
                 t))
            (:verify-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :verify-object)
                 (send *ti* :verify-object *arm*)))
            (:check-picking-fail-count
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-picking-fail-count)
                 (send *ti* :check-picking-fail-count *arm*)))
            (:set-target-bin
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :set-target-bin)
                 (send *ti* :set-target-bin *arm*)))
            (:return-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :return-object)
                 (send *ti* :return-object *arm*)
                 t))
            (:place-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :place-object)
                 (send *ti* :place-object *arm*)
                 t))
            (:return-from-place-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :return-from-place-object)
                 (send *ti* :return-from-place-object *arm*)
                 t))
            (:wait-for-opposite-arm-start-picking
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :wait-for-opposite-arm-start-picking)
                 (send *ti* :check-can-start *arm* :wait-for-opposite-arm
                       :wait-for-opposite-arm-start-picking))))
          '(:init)
          '(:finish))))


(defun stow-init (&key (ctype :default-controller) (calib-pressure t) (moveit nil))
  (jsk_arc2017_baxter::stow-init :ctype ctype :moveit moveit)
  (when moveit (send *ti* :wipe-all-scene))
  (send *ri* :gripper-servo-on)
  ;; initialize fingers
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pre-pose :rarm :cylindrical) 1000 :wait nil)
  (send *ri* :move-hand :larm
        (send *baxter* :hand-grasp-pre-pose :larm :cylindrical) 1000)
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pre-pose :rarm :opposed) 1000 :wait nil)
  (send *ri* :move-hand :larm
        (send *baxter* :hand-grasp-pre-pose :larm :opposed) 1000)
  (send *ri* :angle-vector (send *baxter* :fold-pose-back))
  (send *ri* :wait-interpolation)
  (when calib-pressure
    (send *ri* :calib-pressure-threshold :rarm)
    (send *ri* :calib-pressure-threshold :larm))
  (objects (list *baxter*))
  t)

(defun stow-mainloop (arm)
  (setq *arm* arm)
  (when (not (boundp '*sm*))
    (make-stowing-state-machine))
  (exec-state-machine *sm* nil :root-name (format nil "SM_~a_ROOT" (string-upcase (arm2str arm))) :hz 2.0))

(warn "~% Commands ~%")
(warn "(stow-init)           : initialize *ti*~%")
(warn "(stow-mainloop :rarm) : start the mainloop~%~%")
