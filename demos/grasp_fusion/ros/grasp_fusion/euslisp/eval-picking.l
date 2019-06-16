#!/usr/bin/env roseus

(ros::roseus "robot_main")

(require "package://grasp_fusion/euslisp/lib/stow-interface.l")
;; smach
(require :state-machine "package://roseus_smach/src/state-machine.l")
(require :state-machine-ros "package://roseus_smach/src/state-machine-ros.l")
(require :state-machine-utils "package://roseus_smach/src/state-machine-utils.l")

(make-random-state t)

(defun make-eval-picking-state-machine ()
  (setq *sm*
        (make-state-machine
          '((:init -> :recognize-bboxes)
            (:recognize-bboxes -> :wait-for-user-input)
            (:wait-for-user-input -> :recognize-object)
            (:wait-for-user-input !-> :finish)
            (:recognize-object -> :check-can-start-picking)
            (:recognize-object !-> :return-from-recognize-object)
            (:return-from-recognize-object -> :init)
            (:check-can-start-picking -> :pick-object)
            (:check-can-start-picking !-> :check-start-picking-fail-count)
            (:check-start-picking-fail-count -> :return-from-pick-object)
            (:check-start-picking-fail-count !-> :recognize-object)
            (:pick-object -> :verify-object)
            (:pick-object !-> :change-grasp-style)
            (:change-grasp-style -> :pick-object)
            (:change-grasp-style !-> :return-from-pick-object)
            (:verify-object -> :check-grasp-stability)
            (:verify-object !-> :return-object)
            (:return-from-pick-object -> :calib-prismatic-joint)
            (:check-grasp-stability -> :return-object)
            (:return-object -> :calib-prismatic-joint)
            (:calib-prismatic-joint -> :init)
            (:calib-prismatic-joint !-> :calib-prismatic-joint))
          '((:init
              '(lambda (userdata)
                 ;; initialize slots
                 (send *ti* :init :moveit *moveit*)
                 ;; FIXME: dirty hack
                 (if (boundp '*primitives*)
                   (send *ti* :set-val 'primitives- *primitives*))
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
            (:recognize-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :recognize-object)
                 (send *ti* :recognize-object *arm* :trial-times 3 :use-euc-cl *use-euc-cl*)))
            (:return-from-recognize-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :return-from-recognize-object)
                 (send *ti* :return-from-recognize-object *arm*)
                 t))
            (:check-can-start-picking
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-can-start-picking)
                 (send *ti* :check-can-start-picking *arm*)))
            (:check-start-picking-fail-count
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-start-picking-fail-count)
                 (send *ti* :check-start-picking-fail-count *arm* :count-limit 5)))
            (:pick-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :pick-object)
                 (send *ti* :pick-object *arm*
                       :use-scale t :move-head t :avoid-tote t :lift-full *lift-full*)))
            (:change-grasp-style
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :change-grasp-style)
                 (send *ti* :change-grasp-style *arm*)))
            (:return-from-pick-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :return-from-pick-object)
                 (send *ti* :return-from-pick-object *arm*)
                 t))
            (:verify-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :verify-object)
                 (send *ti* :verify-object *arm*)))
            (:check-grasp-stability
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-grasp-stability)
                 (dolist (axis (list :y :z))
                   (send *ti* :spin-off-around-axis *arm* axis :times 5)
                   (send *ri* :wait-interpolation)
                   (unix::sleep 1))
                 t))
            (:return-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :return-object)
                 (send *ti* :return-object *arm*)
                 t))
            (:calib-prismatic-joint
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :calib-prismatic-joint)
                 (send *ti* :calib-prismatic-joint *arm*))))
          '(:init)
          '(:finish))))


(defun eval-picking-init (&key (ctype :default-controller) (calib-pressure t) (moveit nil))
  (setq *moveit* moveit)
  (setq *lift-full* t)
  (setq *use-euc-cl* nil)
  (grasp_fusion::stow-init :ctype ctype :moveit moveit)
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
  (send *baxter* :arc-reset-pose)
  (send *baxter* :head_pan :joint-angle -70)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)
  (when calib-pressure
    (send *ri* :calib-pressure-threshold :rarm)
    (send *ri* :calib-pressure-threshold :larm))
  (objects (list *baxter*))
  t)

(defun eval-picking-mainloop (arm)
  (setq *arm* arm)
  (when (not (boundp '*sm*))
    (make-eval-picking-state-machine))
  (exec-state-machine *sm* nil :root-name (format nil "SM_~a_ROOT" (string-upcase (arm2str arm))) :hz 2.0))

(defun set-ignore-ins (arm ignore-ins)
  (let ((res (ros::service-call
               (format nil "~a_hand_camera/primitive_matching/ignore_instance"
                       (arm2str arm))
               (instance std_srvs::SetBoolRequest :init :data ignore-ins))))
    (if (send res :success)
      (ros::ros-info-green "Setting ignore_instance succeeded")
      (ros::ros-error "Setting ignore_instance failed"))))

(warn "~% Commands ~%")
(warn "(eval-picking-init)                 : initialize *ti*~%")
(warn "(setq *primitives* (list :suction)) : (OPTIONAL) restrict primitive choices~%")
(warn "(setq *lift-full* nil)              : (OPTIONAL) don't lift high for fragile object~%")
(warn "(set-ignore-ins :rarm t)            : (OPTIONAL) ignore instance segmentation~%")
(warn "(setq *use-euc-cl* t)               : (OPTIONAL) use euclidean clustering instead of primitive matching~%")
(warn "(eval-picking-mainloop :rarm)       : start the mainloop~%~%")
