#!/usr/bin/env roseus

(ros::roseus "robot_main")
(require "package://dualarm_grasping/euslisp/lib/dualarm-grasp-interface.l")
;; smach
(require :state-machine "package://roseus_smach/src/state-machine.l")
(require :state-machine-ros "package://roseus_smach/src/state-machine-ros.l")
(require :state-machine-utils "package://roseus_smach/src/state-machine-utils.l")

(make-random-state t)


(defun make-dualarm-grasp-state-machine ()
  (setq *sm*
        (make-state-machine
          '((:init -> :recognize-bboxes)
            (:recognize-bboxes -> :wait-for-user-input)
            (:wait-for-user-input -> :recognize-object)
            (:wait-for-user-input !-> :finish)
            (:recognize-object -> :pick-object)
            (:recognize-object !-> :recognize-object)
            (:recognize-object :give-up :wait-for-user-input)
            (:pick-object -> :verify-object)
            (:pick-object !-> :recognize-object)
            (:verify-object -> :place-object)
            (:verify-object !-> :return-object)
            (:verify-object :no-target :remove-object)
            (:place-object -> :recognize-object)
            (:return-object -> :recognize-object)
            (:remove-object -> :recognize-object))
          '((:init
              '(lambda (userdata)
                 (send *ti* :reset-giveup :larm)
                 (send *ti* :reset-giveup :rarm)
                 (ros::ros-info-green "[dualarm-grasp] start picking")
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
                       success-p)
                   (setq arm (send *ti* :set-recognize-arm arm :update t))
                   (unix::sleep 1)
                   (setq success-p
                         (send *ti* :recognize-object arm))
                   (if success-p
                     (progn
                       (send *ti* :set-fail-recognize-arm nil)
                       (send *ti* :set-save-dir)
                       (send *ti* :recognition-save-request))
                     (progn
                       ; (send *ti* :giveup)
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
                   (setq success-p (send *ti* :graspingp arm))
                   (unless success-p
                     (send *ri* :stop-grasp arm)
                     (ros::ros-info-green "[dualarm-grasp] pick failed because of ~A" pick-result)
                     (send *ti* :return-from-pick-object arm)
                     ; (send *ti* :giveup)
                     )
                   (send *ti* :set-result pick-result)
                   (send *ti* :result-save-request)
                   (if (or (eq pick-result :moveit-failed) (eq pick-result :ik-failed))
                     (send *ti* :set-fail-recognize-arm (send *ti* :get-recognize-arm)))
                   success-p)))
            (:verify-object
              '(lambda (userdata)
                 (ros::ros-info-green "[dualarm-grasp] is it target?: ~A" (send *ti* :get-is-target))
                 (if (send *ti* :verify-object)
                   (if (send *ti* :get-is-target) t :no-target)
                   nil)))
            (:place-object
              '(lambda (userdata)
                 (let ((arm (cdr (assoc 'grasp-arm userdata)))
                       (target-grasp (cdr (assoc 'target-grasp userdata))))
                   (if target-grasp
                     (send *ti* :place-object arm :distance 300)
                     (send *ti* :place-object arm))
                   (send *ti* :return-from-place-object arm)
                   (send *ti* :calib-prismatic-joint arm))
                 t))
            (:return-object
              '(lambda (userdata)
                 (let ((arm (cdr (assoc 'grasp-arm userdata))))
                   (send *ti* :return-object arm)
                   (send *ti* :return-from-pick-object arm)
                   (send *ti* :calib-prismatic-joint arm)
                   ; (send *ti* :giveup)
                   )
                 t))
            (:remove-object
              '(lambda (userdata)
                 (let ((arm (cdr (assoc 'grasp-arm userdata)))
                       (target-grasp (cdr (assoc 'target-grasp userdata))))
                   (if target-grasp
                     (send *ti* :place-object arm :opposite t :distance 300)
                     (send *ti* :place-object arm :opposite t))
                   (send *ti* :return-from-place-object arm)
                   (send *ti* :calib-prismatic-joint arm))
                 t)))
          '(:init)
          '(:finish))))


(defun dualarm-grasp-init (&key (ctype :default-controller) (calib-pressure t)
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
  (objects (list *baxter*))
  t)


(defun dualarm-grasp-mainloop (&key (target-grasp nil))
  (when (not (boundp '*sm*))
    (make-dualarm-grasp-state-machine))
  (exec-state-machine *sm* `((grasp-arm . :larm)
                             (target-grasp . ,target-grasp)) :hz 2.0))


(warn "~% Commands ~%")
(warn "(dualarm-grasp-init)                     : initialize *ti*~%")
(warn "(dualarm-grasp-mainloop)                 : start random grasping mainloop~%")
(warn "(dualarm-grasp-mainloop :target-grasp t) : start target grasping mainloop~%")
