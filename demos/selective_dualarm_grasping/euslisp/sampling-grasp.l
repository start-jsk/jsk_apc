#!/usr/bin/env roseus

(ros::roseus "robot_main")
(require "package://dualarm_grasping/euslisp/lib/dualarm-grasp-interface.l")
;; smach
(require :state-machine "package://roseus_smach/src/state-machine.l")
(require :state-machine-ros "package://roseus_smach/src/state-machine-ros.l")
(require :state-machine-utils "package://roseus_smach/src/state-machine-utils.l")

(make-random-state t)


(defun make-sampling-grasp-state-machine ()
  (setq *sm*
        (make-state-machine
          '((:init -> :recognize-bboxes)
            (:recognize-bboxes -> :wait-for-user-input)
            (:wait-for-user-input -> :recognize-object)
            (:wait-for-user-input !-> :finish)
            (:recognize-object -> :pick-object)
            (:recognize-object !-> :recognize-object)
            (:recognize-object :give-up :wait-for-user-input)
            (:pick-object -> :return-object)
            (:pick-object !-> :recognize-object)
            (:return-object -> :recognize-object))
          '((:init
              '(lambda (userdata)
                 (ros::ros-info-green "[sampling-grasp] start picking")
                 t))
            (:recognize-bboxes
              '(lambda (userdata)
                 (send *ti* :recognize-bboxes)
                 t))
            (:wait-for-user-input
              '(lambda (userdata)
                 (send *ti* :wait-for-user-input)))
            (:recognize-object
              '(lambda (userdata)
                 (let ((arm (elt '(:larm :rarm) (random 2)))
                       (grasping-way (cdr (assoc 'grasping-way userdata)))
                       success-p)
                   (setq arm (send *ti* :set-recognize-arm arm :update t))
                   (send *ti* :set-grasping-way grasping-way)
                   (unix::sleep 1)
                   (setq success-p
                         (send *ti* :recognize-object arm))
                   (if success-p
                     (progn
                       (send *ti* :set-fail-recognize-arm nil)
                       (send *ti* :set-save-dir)
                       (send *ti* :recognition-save-request))
                     (progn
                       (send *ti* :set-fail-recognize-arm arm)
                       (send *ti* :return-from-recognize-object arm)))
                   (setf (cdr (assoc 'grasp-arm userdata))
                         (send *ti* :get-grasp-arm arm))
                   success-p)))
            (:pick-object
              '(lambda (userdata)
                 (let (pick-result success-p (arm (cdr (assoc 'grasp-arm userdata))))
                   (send *ti* :calib-prismatic-joint arm)
                   (setq pick-result (send *ti* :pick-object arm))
                   (unix::sleep 1)
                   (setq success-p (send *ti* :graspingp arm))
                   (if success-p
                     (ros::ros-info-green "[sampling-grasp] pick succeeded")
                     (progn
                       (when (eq pick-result :grasp-succeeded) (setq pick-result :grasp-unstable))
                       (ros::ros-error "[sampling-grasp] pick failed because of ~A" pick-result)))
                   (send *ti* :set-result pick-result)
                   (send *ti* :result-save-request)
                   (unless success-p
                     (send *ri* :stop-grasp arm)
                     (send *ti* :return-from-pick-object arm))
                   success-p)))
            (:return-object
              '(lambda (userdata)
                 (let ((arm (cdr (assoc 'grasp-arm userdata))))
                   (send *ti* :return-object arm)
                   (send *ti* :return-from-pick-object arm)
                   (send *ti* :calib-prismatic-joint arm))
                 t)))
          '(:init)
          '(:finish))))


(defun sampling-grasp-init (&key (ctype :default-controller) (calib-pressure t)
                                 (moveit t) (scale nil))
  (dualarm_grasping::dualarm-grasp-init :ctype ctype :moveit moveit :scale scale)
  (when moveit
    (send *ti* :wipe-all-scene)
    (send *ti* :add-workspace-scene))
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
  (send *ti* :calib-prismatic-joint :arms)
  (objects (list *baxter*))
  t)


(defun sampling-grasp-mainloop (gway)
  (when (not (boundp '*sm*))
    (make-sampling-grasp-state-machine))
  (exec-state-machine *sm* `((grasp-arm . :larm)
                             (grasping-way . ,(if (eq gway :dual) "dual" "single"))) :hz 2.0))


(warn "~% Commands ~%")
(warn "(sampling-grasp-init)             : initialize *ti*~%")
(warn "(sampling-grasp-mainloop :single) : start single arm sampling mainloop~%")
(warn "(sampling-grasp-mainloop :dual)   : start dual arm sampling mainloop~%")
