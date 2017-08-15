#!/usr/bin/env roseus
;; vim: set ft=lisp:

(ros::load-ros-manifest "jsk_arc2017_baxter")

(defun remember-messages (key msg)
  (sethash key *last-messages* msg))

(ros::roseus "check_gripper_state")
(setq *start-time* (ros::time-now))
(setq *last-messages* (make-hash-table))
(setq *message-keys* (list :finger-tendon :finger-yaw :prismatic-joint :vacuum-pad-tendon))
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

(setq *target-node-name* (ros::get-param "~target_node_name"))
(ros::rate 1)
(ros::ros-info "check if ~A alive..." *target-node-name*)
(while (ros::ok)
  (ros::spin-once)
  (dolist (key *message-keys*)
    (let (time-elapsed msg timeoutp)
      (setq msg (gethash key *last-messages*))
      (if msg
        (progn
          (setq time-elapsed (- (send (ros::time-now) :to-sec)
                                (send (send msg :header :stamp) :to-sec)))
          (ros::ros-debug "  ~A received ~A [sec] ago" (send msg :name) time-elapsed)
          (when (> time-elapsed 1)
            (ros::ros-error "  ~A received ~A [sec] ago, kill ~A"
                            (send msg :name) time-elapsed *target-node-name*)
            (setq timeoutp t)))
        (progn
          (setq time-elapsed (- (send (ros::time-now) :to-sec) (send *start-time* :to-sec)))
          (ros::ros-debug "  ~A never received in ~A [sec]" key time-elapsed)
          (when (> time-elapsed 5)
            (ros::ros-error "  ~A never received in ~A [sec], kill ~A"
                            key time-elapsed *target-node-name*)
            (setq timeoutp t))))
      (when timeoutp
        (unix::system (format nil "rosnode kill ~A" *target-node-name*))
        (setq namespace (subseq (ros::get-namespace) 1))
        (ros::ros-error "  restart swpaner ~A" namespace)
        (unix::system (format nil "ROS_NAMESPACE=  roslaunch jsk_arc2017_baxter gripper_v6_dxl_controller.launch --args ~A/controller_spawner | bash" namespace))
        (ros::exit 1))))
  (ros::sleep))
