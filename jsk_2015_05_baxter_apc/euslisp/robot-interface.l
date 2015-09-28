;; -*- mode: lisp;-*-

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")

(defmethod baxter-interface
  ;; ----------------------
  ;; Control vacuum gripper
  ;; ----------------------
  (:start-grasp
   (&optional (arm :arms))
   (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm-to-str arm))))
     (ros::advertise-if-yet (format nil "/vacuum_gripper/limb/~A" l/r) std_msgs::Bool 1)
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
     (ros::advertise-if-yet (format nil "/vacuum_gripper/limb/~A" l/r) std_msgs::Bool 1)
     (setq msg (instance std_msgs::Bool :init))
     (send msg :data nil)
     (ros::ros-info (format nil "stop ~A vacuum gripper" l/r))
     (ros::publish (format nil "/vacuum_gripper/limb/~A" l/r) msg)
     (ros::sleep)
     ))
  (:arm-potentio-vector
    (arm)
    (case arm
      (:larm (subseq (send *ri* :state :potentio-vector) 1 8))
      (:rarm (subseq (send *ri* :state :potentio-vector) 8 15))
      (t nil)
      )
    )
  )
