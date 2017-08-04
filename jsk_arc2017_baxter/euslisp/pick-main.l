#!/usr/bin/env roseus

(ros::roseus "robot_main")

(require "package://jsk_arc2017_baxter/euslisp/lib/pick-interface.l")
;; smach
(require :state-machine "package://roseus_smach/src/state-machine.l")
(require :state-machine-ros "package://roseus_smach/src/state-machine-ros.l")
(require :state-machine-utils "package://roseus_smach/src/state-machine-utils.l")

(make-random-state t)

(defun make-picking-state-machine ()
  (setq *sm*
        (make-state-machine
          '((:init -> :recognize-bboxes)
            (:recognize-bboxes -> :wait-for-user-input)
            (:wait-for-user-input -> :wait-for-opposite-arm)
            (:wait-for-user-input !-> :finish)
            (:wait-for-opposite-arm -> :set-target)
            (:wait-for-opposite-arm !-> :wait-for-opposite-arm)
            (:set-target -> :recognize-object)
            (:set-target !-> :wait-for-user-input)
            (:recognize-object -> :check-can-start-picking)
            (:recognize-object !-> :return-from-recognize-object)
            (:return-from-recognize-object -> :check-recognize-fail-count)
            (:check-recognize-fail-count -> :wait-for-user-input)
            (:check-recognize-fail-count !-> :wait-for-opposite-arm-start-picking)
            (:check-can-start-picking -> :pick-object)
            (:check-can-start-picking !-> :check-start-picking-fail-count)
            (:check-start-picking-fail-count -> :return-from-pick-object)
            (:check-start-picking-fail-count !-> :recognize-object)
            (:pick-object -> :verify-object)
            (:pick-object !-> :change-grasp-style)
            (:change-grasp-style -> :pick-object)
            (:change-grasp-style !-> :return-from-pick-object)
            (:return-from-pick-object -> :check-picking-fail-count)
            (:check-picking-fail-count -> :wait-for-opposite-arm-start-picking)
            (:check-picking-fail-count !-> :calib-prismatic-joint)
            (:verify-object -> :set-target-cardboard)
            (:verify-object !-> :return-object)
            (:return-object -> :check-picking-fail-count)
            (:set-target-cardboard -> :place-object)
            (:set-target-cardboard !-> :return-object)
            (:place-object -> :return-from-place-object)
            (:return-from-place-object -> :calib-prismatic-joint)
            (:wait-for-opposite-arm-start-picking -> :calib-prismatic-joint)
            (:wait-for-opposite-arm-start-picking !-> :wait-for-opposite-arm-start-picking)
            (:calib-prismatic-joint -> :wait-for-opposite-arm)
            (:calib-prismatic-joint !-> :calib-prismatic-joint))
          '((:init
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :init)
                 (ros::ros-info "start ~a picking" (arm2str *arm*))
                 t))
            (:recognize-bboxes
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :recognize-bboxes)
                 (send *ti* :recognize-bboxes)
                 t))
            (:wait-for-user-input
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :wait-for-user-input)
                 (send *ti* :wait-for-user-input *arm*)))
            (:wait-for-opposite-arm
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :wait-for-opposite-arm)
                 (send *ti* :check-can-start *arm* :set-target
                       :wait-for-opposite-arm)))
            (:set-target
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :set-target)
                 (send *ti* :set-target *arm*)))
            (:recognize-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :recognize-object)
                 (send *ti* :recognize-object *arm* :trial-times 3)))
            (:return-from-recognize-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :return-from-recognize-object)
                 (send *ti* :return-from-recognize-object *arm*)
                 t))
            (:check-recognize-fail-count
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-recognize-fail-count)
                 (send *ti* :check-recognize-fail-count *arm* :count-limit 10)))
            (:check-can-start-picking
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-can-start-picking)
                 (send *ti* :check-can-start-picking *arm*)))
            (:check-start-picking-fail-count
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-start-picking-fail-count)
                 (send *ti* :check-start-picking-fail-count *arm* :count-limit 5)))
            (:change-grasp-style
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :change-grasp-style)
                 (send *ti* :change-grasp-style *arm*)))
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
            (:set-target-cardboard
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :set-target-cardboard)
                 (send *ti* :set-target-cardboard *arm*)))
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
                       :wait-for-opposite-arm-start-picking)))
            (:calib-prismatic-joint
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :calib-prismatic-joint)
                 (send *ti* :calib-prismatic-joint *arm*))))
          '(:init)
          '(:finish))))


(defun pick-init (&key (ctype :default-controller) (calib-pressure t) (moveit nil))
  (jsk_arc2017_baxter::pick-init :ctype ctype :moveit moveit)
  (when moveit
    (send *ti* :wipe-all-scene)
    (send *ti* :add-workspace-scene)
    )
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
  (send *ri* :angle-vector (send *baxter* :arc-reset-pose))
  (send *ri* :wait-interpolation)
  (when calib-pressure
    (send *ri* :calib-pressure-threshold :rarm)
    (send *ri* :calib-pressure-threshold :larm))
  (objects (list *baxter*))
  t)

(defun pick-mainloop (arm)
  (setq *arm* arm)
  (when (not (boundp '*sm*))
    (make-picking-state-machine))
  (exec-state-machine *sm* nil :root-name (format nil "SM_~a_ROOT" (string-upcase (arm2str arm))) :hz 2.0))

(warn "~% Commands ~%")
(warn "(pick-init)           : initialize *ti*~%")
(warn "(pick-mainloop :rarm) : start the mainloop~%~%")
