#!/usr/bin/env roseus
;; vim: set ft=lisp:

(ros::load-ros-manifest "jsk_arc2017_baxter")

(defun remember-messages (type msg)
  (sethash type *last-messages* msg))

(ros::roseus "check_gripper_state")
(setq *last-messages* (make-hash-table))
(ros::subscribe "finger_tendon_controller/state"
                dynamixel_msgs::JointState
                #'remember-messages :finger-tendon 1)
(ros::subscribe "finger_yaw_joint_controller/state"
                dynamixel_msgs::JointState
                #'remember-messages :finger-yaw 1)
(ros::subscribe "prismatic_joint_controller/state"
                dynamixel_msgs::JointState
                #'remember-messages :prismatic-joint 1)
(ros::subscribe "vacuum_pad_tendon_controller/state"
                dynamixel_msgs::JointState
                #'remember-messages :vacuum-pad-tendon 1)

(setq target-node-name (ros::get-param "~target_node_name"))
(ros::rate 1)
(ros::ros-info "check if ~A alive..." target-node-name)
(while (ros::ok)
  (ros::spin-once)
  (dolist (msg (send *last-messages* :list-values))
    (setq time-elapsed (- (send (ros::time-now) :to-sec) (send (send msg :header :stamp) :to-sec)))
    ;(ros::ros-info "  ~A received ~A [sec] ago" (send msg :name) time-elapsed)
    (when (>  time-elapsed 1)
      (ros::ros-error "  ~A received ~A [sec] ago, kill ~A" (send msg :name) time-elapsed target-node-name)
      (unix::system (format nil "rosnode kill ~A" target-node-name))
      (setq namespace (subseq (ros::get-namespace) 1))
      (ros::ros-error "  restart swpaner ~A" namespace)
      (unix::system (format nil "ROS_NAMESPACE=  roslaunch jsk_arc2017_baxter gripper_v6_dxl_controller.launch --args ~A/controller_spawner | bash" namespace))
      (ros::exit 1)))
  (ros::sleep))
