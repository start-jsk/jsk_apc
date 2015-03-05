#!/usr/bin/env roseus
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")
(load "package://jsk_2014_picking_challenge/euslisp/gripper-control.l")

;; (defmethod baxter-interface
;;   (:start-grasp ()
;;    (ros::ros-info (format nil "cleaner ON"))
;;    ;;(ros::publish "on_off_gripper" (instance std_msgs::Empty :init))
;;    (setq msg (instance std_msgs::Bool :init))
;;    (send msg :data t)
;;    (ros::publish "on_off_gripper" msg)
;;    (ros::sleep))
;;   (:stop-grasp ()
;;    (ros::ros-info (format nil "cleaner OFF"))
;;    ;;(ros::publish "on_off_gripper" (instance std_msgs::Empty :init))
;;    (setq msg (instance std_msgs::Bool :init))
;;    (send msg :data nil)
;;    (ros::publish "on_off_gripper" msg)
;;    (ros::sleep)
;;   ))

(defun init ()
  (baxter-init)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *baxter* :locate #f(0 0 950) :world)
  (objects (list *baxter*)))

(defun start ()
  (send *ri* :start-grasp))
(defun stop ()
  (send *ri* :stop-grasp))

(init)
