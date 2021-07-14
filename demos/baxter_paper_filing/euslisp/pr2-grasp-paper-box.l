(load "package://pr2eus/pr2-interface.l")


(ros::load-ros-package "pr2_gripper_sensor_msgs")

(setq *init-av*
      #f(11.5 17.011 -20.2579 74.8403 -79.0273 72.1499 -103.437 286.197 -39.9069
         56.9961 6.21242 -113.603 5.49354 -27.6086 -443.792 6.7387 -21.2682))

;; http://wiki.ros.org/pr2_gripper_sensor_action
(defun l-gripper-sensor-controller-init ()
  ;; Set the same parameter as https://ieeexplore.ieee.org/document/6224985?tp=&arnumber=6224985
  (ros::set-param "/l_gripper_sensor_controller/force_lightest" 0.4)
  (ros::set-param "/l_gripper_sensor_controller/hp_force_trigger" 0.05)
  (unix::sleep 1)  ;; just in case
  (call-empty-service "/l_gripper_sensor_controller/reload_params")
  (call-empty-service "/l_gripper_sensor_controller/zero_fingertip_sensors"))

(defun grasp-paper-box-init (&key (inst nil))
  (when inst
    (pr2-init t)
    (setq *grab-action* (instance ros::simple-action-client :init
                                  "/l_gripper_sensor_controller/grab"
                                  pr2_gripper_sensor_msgs::PR2GripperGrabAction))
    (setq *release-action* (instance ros::simple-action-client :init
                                     "/l_gripper_sensor_controller/release"
                                     pr2_gripper_sensor_msgs::PR2GripperReleaseAction))
    (send *grab-action* :wait-for-server)
    (send *release-action* :wait-for-server))
  (send *ri* :angle-vector (send *pr2* :angle-vector *init-av*) 5000)
  (send *release-action* :send-goal (send *release-action* :make-goal-instance))
  (send *ri* :wait-interpolation))
(defun grasp-paper-box ()
  (l-gripper-sensor-controller-init)
  (send *ri* :angle-vector
        (send *pr2* :larm :move-end-pos (float-vector 0 0 -90) :world) 3000)
  (send *ri* :wait-interpolation)
  (unix::sleep 1)
  (send *grab-action* :send-goal (send *grab-action* :make-goal-instance))
  (send *grab-action* :wait-for-result))
