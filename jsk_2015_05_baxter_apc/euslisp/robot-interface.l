;; -*- mode: lisp;-*-

(load "package://baxtereus/baxter-interface.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")

(defmethod baxter-interface
  ;; ----------------------
  ;; Control vacuum gripper
  ;; ----------------------
  (:start-grasp
   (&optional (arm :arms))
   (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm-to-str arm))))
     (unless (ros::get-topic-publisher (format nil "/vacuum_gripper/limb/~A" (arm-to-str arm)))
       (ros::advertise (format nil "/vacuum_gripper/limb/~A" l/r) std_msgs::Bool 1)
       (unix:sleep 1))
     (setq msg (instance std_msgs::Bool :init))
     (send msg :data t)
     (ros::ros-info (format nil "start ~A vacuum gripper" l/r))
     (ros::publish (format nil "/vacuum_gripper/limb/~A" l/r) msg)
     (ros::sleep)
     ))
  (:go-grasp (&optional (arm :arms)) (send self :start-grasp arm))
  (:stop-grasp
   (&optional (arm :arms))
   (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm-to-str arm))))
     (unless (ros::get-topic-publisher (format nil "/vacuum_gripper/limb/~A" (arm-to-str arm)))
       (ros::advertise (format nil "/vacuum_gripper/limb/~A" l/r) std_msgs::Bool 1)
       (unix:sleep 1))
     (setq msg (instance std_msgs::Bool :init))
     (send msg :data nil)
     (ros::ros-info (format nil "stop ~A vacuum gripper" l/r))
     (ros::publish (format nil "/vacuum_gripper/limb/~A" l/r) msg)
     (ros::sleep)
     ))
  )
