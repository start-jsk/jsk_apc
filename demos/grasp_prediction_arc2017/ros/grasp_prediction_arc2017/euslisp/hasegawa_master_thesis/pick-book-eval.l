#!/usr/bin/env roseus

(ros::roseus "robot_main")

(require "package://grasp_prediction_arc2017/euslisp/hasegawa_master_thesis/lib/pick-book-interface.l")
;; smach
(require :state-machine "package://roseus_smach/src/state-machine.l")
(require :state-machine-ros "package://roseus_smach/src/state-machine-ros.l")
(require :state-machine-utils "package://roseus_smach/src/state-machine-utils.l")

(make-random-state t)

(defun make-eval-state-machine ()
  (setq *sm*
        (make-state-machine
          '((:init -> :recognize-bboxes)
            (:recognize-bboxes -> :wait-for-user-input)
            (:wait-for-user-input -> :set-target)
            (:wait-for-user-input !-> :finish)
            (:set-target -> :recognize-object)
            (:set-target !-> :init)
            (:recognize-object -> :check-can-start-picking)
            (:recognize-object !-> :return-from-recognize-object)
            (:return-from-recognize-object -> :init)
            (:check-can-start-picking -> :pick-object)
            (:check-can-start-picking !-> :check-start-picking-fail-count)
            (:check-start-picking-fail-count -> :return-from-pick-object)
            (:check-start-picking-fail-count !-> :recognize-object)
            (:pick-object -> :verify-object)
            (:pick-object !-> :check-tried-directions)
            (:return-from-pick-object -> :calib-prismatic-joint)
            (:verify-object -> :check-grasp-stability)
            (:verify-object !-> :return-object)
            (:check-tried-directions -> :recognize-object)
            (:check-tried-directions !-> :return-from-pick-object)
            (:check-grasp-stability -> :return-object)
            (:return-object -> :calib-prismatic-joint)
            (:calib-prismatic-joint -> :init)
            (:calib-prismatic-joint !-> :calib-prismatic-joint))
          '((:init
              '(lambda (userdata)
                 ;; initialize slots
                 (send *ti* :init :moveit *moveit*)
                 (send *ti* :update-state *arm* :init)
                 (if (not *ignore-suction-pt*)
                   (setq *skip-dirs* nil)
                   (setq *skip-dirs*
                         (list
                           (list #f(0 1 0) #f(-1 0 0) #f(0 -1 0))
                           (list #f(1 0 0) #f(-1 0 0) #f(0 -1 0))
                           (list #f(1 0 0) #f(0 1 0) #f(0 -1 0))
                           ;; uncomment the following if you want to try correct edge at last
                           ;; (list #f(1 0 0) #f(0 1 0) #f(-1 0 0))
                           )))
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
            (:set-target
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :set-target)
                 (send *ti* :set-target *arm*)))
            (:recognize-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :recognize-object)
                 (send *ti* :recognize-object *arm* :trial-times 10)))
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
            (:check-tried-directions
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :check-tried-directions)
                 (when *skip-dirs*
                   (send *ti* :set-val 'tried-directions- (pop *skip-dirs*)))
                 (send *ti* :check-tried-directions *arm*)))
            (:pick-object
              '(lambda (userdata)
                 (send *ti* :update-state *arm* :pick-object)
                 ;; uncomment the following if you want to try correct edge at last
                 ;; (when (= (length *skip-dirs*) 4)
                 (when (= (length *skip-dirs*) 3)
                   (send *ti* :set-val 'tried-directions- (pop *skip-dirs*)))
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


(defun pick-book-eval-init (&key (ctype :default-controller) (calib-pressure t) (moveit nil))
  (setq *moveit* moveit)
  (grasp_prediction_arc2017::pick-book-init :ctype ctype :moveit moveit)
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

(defun pick-book-eval-mainloop (arm &key (ignore-suction-pt nil))
  (setq *arm* arm)
  (setq *ignore-suction-pt* ignore-suction-pt)
  (when (not (boundp '*sm*))
    (make-eval-state-machine))
  (exec-state-machine *sm* nil :root-name (format nil "SM_~a_ROOT" (string-upcase (arm2str arm))) :hz 2.0))

(warn "~% Commands ~%")
(warn "(pick-book-eval-init :ctype :larm-head-controller :moveit t) : initialize *ti*~%")
(warn "(pick-book-eval-mainloop :larm)                              : start the mainloop~%")
(warn "(pick-book-eval-mainloop :larm :ignore-suction-pt t)         : start the mainloop ignoring suction point~%~%")
