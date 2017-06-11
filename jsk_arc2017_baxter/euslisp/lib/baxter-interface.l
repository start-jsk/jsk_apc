;; -*- mode: lisp;-*-

(require "package://baxtereus/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")


(ros::load-ros-manifest "jsk_arc2017_baxter")

(unless (find-package "JSK_ARC2017_BAXTER")
  (make-package "JSK_ARC2017_BAXTER"))

(defclass jsk_arc2017_baxter::baxter-interface
  :super baxter-interface
  :slots (rarm-pressure-threshold-
          larm-pressure-threshold-))

(defmethod jsk_arc2017_baxter::baxter-interface
  (:init
    (&rest args &key
           ((:moveit-environment mvit-env)
            (instance jsk_2016_01_baxter_apc::baxter-moveit-environment))
           ((:moveit-robot mvit-rb) (instance jsk_2016_01_baxter_apc::baxter-robot :init))
           &allow-other-keys)
    ;; initialize controllers
    (send-super* :init :joint-states-queue-size 2
                 ;; FIXME: :robot jsk_arc2017_baxter::baxter-robot
                 ;; FIXME: :end-coords-interpolation does not work.
                 :robot baxter-robot
                 :moveit-environment nil
                 :moveit-robot nil args)
    (send self :add-controller :no-gripper-controller)
    (send self :add-controller :rarm-no-gripper-controller)
    (send self :add-controller :larm-no-gripper-controller)
    (send self :add-controller :rgripper-controller)
    (send self :add-controller :lgripper-controller)
    ;; hack for https://github.com/jsk-ros-pkg/jsk_pr2eus/issues/227
    (if (not (equal (send (car (gethash :rarm-controller (self . controller-table))) :name)
                    (cdr (assoc :controller-action (car (send self :rarm-controller))))))
        (let ((tmp-actions (gethash :rarm-controller controller-table)))
          (setf (gethash :rarm-controller controller-table) (reverse tmp-actions))))
    (if (not (equal (send (car (gethash :larm-controller (self . controller-table))) :name)
                    (cdr (assoc :controller-action (car (send self :larm-controller))))))
        (let ((tmp-actions (gethash :larm-controller controller-table)))
          (setf (gethash :larm-controller controller-table) (reverse tmp-actions))))
    ;; initialize slots
    (unless (ros::get-param "/apc_on_gazebo" nil)
      (ros::advertise "/vacuum_gripper/limb/left" std_msgs::Bool)
      (ros::advertise "/vacuum_gripper/limb/right" std_msgs::Bool))
    (ros::advertise "/gripper_front/limb/left/servo/torque" std_msgs::Bool)
    (ros::advertise "/gripper_front/limb/right/servo/torque" std_msgs::Bool)
    (if mvit-rb (setq moveit-robot mvit-rb))
    (if mvit-env (send self :set-moveit-environment (send mvit-env :init :robot moveit-robot))))
  ;; Overwrite super class's :rarm-controller
  (:rarm-controller ()
   (append
    (send-super :head-controller)
    (send-super :rarm-controller)
    (send self :rgripper-controller)))
  ;; Overwrite super class's :larm-controller
  (:larm-controller ()
   (append
    (send-super :head-controller)
    (send-super :larm-controller)
    (send self :lgripper-controller)))
  ;; Overwrite super class's :default-controller
  (:default-controller ()
   (append
    (send-super :head-controller)
    (send-super :rarm-controller)
    (send self :rgripper-controller)
    (send-super :larm-controller)
    (send self :lgripper-controller)))
  ;; Rename super class's :rarm-controller to :rarm-no-gripper-controller
  (:rarm-no-gripper-controller ()
   (send-super :rarm-controller))
  ;; Rename super class's :larm-controller to :larm-no-gripper-controller
  (:larm-no-gripper-controller ()
   (send-super :larm-controller))
  (:no-gripper-controller ()
   (append
    (send self :rarm-no-gripper-controller)
    (send self :larm-no-gripper-controller)))
  (:rgripper-controller ()
   (list
    (list
     (cons :controller-action "/gripper_front/limb/right/follow_joint_trajectory")
     (cons :controller-state "/gripper_front/limb/right/state")
     (cons :action-type control_msgs::FollowJointTrajectoryAction)
     (cons :joint-names (list "right_gripper_vacuum_pad_joint")))))
  (:lgripper-controller ()
   (list
    (list
     (cons :controller-action "/gripper_front/limb/left/follow_joint_trajectory")
     (cons :controller-state "/gripper_front/limb/left/state")
     (cons :action-type control_msgs::FollowJointTrajectoryAction)
     (cons :joint-names (list "left_gripper_vacuum_pad_joint")))))
  (:get-arm-controller (arm)
    (cond ((eq arm :rarm) :rarm-controller)
          ((eq arm :larm) :larm-controller)
          (t nil)))
  (:start-grasp
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm2str arm))))
      (if (ros::get-param "/apc_on_gazebo" nil)
        (ros::service-call
          (format nil "/robot/~a_vacuum_gripper/on" l/r)
          (instance std_srvs::EmptyRequest :init))
        (ros::publish
          (format nil "/vacuum_gripper/limb/~a" l/r)
          (instance std_msgs::Bool :init :data t)))))
  (:stop-grasp
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm2str arm))))
      (if (ros::get-param "/apc_on_gazebo" nil)
        (ros::service-call
          (format nil "/robot/~a_vacuum_gripper/off" l/r)
          (instance std_srvs::EmptyRequest :init))
        (ros::publish
          (format nil "/vacuum_gripper/limb/~a" l/r)
          (instance std_msgs::Bool :init :data nil)))))
  (:gripper-servo-on
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm2str arm))))
      (ros::publish
        (format nil "/gripper_front/limb/~a/servo/torque" l/r)
        (instance std_msgs::Bool :init :data t))))
  (:gripper-servo-off
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm2str arm))))
      (ros::publish
        (format nil "/gripper_front/limb/~a/servo/torque" l/r)
        (instance std_msgs::Bool :init :data nil))))
  (:graspingp
    (arm)
    (let (topic)
      (if (ros::get-param "/apc_on_gazebo" nil)
        (progn
          (setq topic (format nil "/robot/~a_vacuum_gripper/grasping" (arm-to-str arm)))
          (send (one-shot-subscribe topic std_msgs::Bool) :data)
          )
        (progn
          (setq topic (format nil "gripper_front/limb/~a/pressure/state" (arm-to-str arm)))
          (< (send (one-shot-subscribe topic std_msgs::Float64) :data)
             (cond
               ((eq arm :rarm) rarm-pressure-threshold-)
               ((eq arm :larm) larm-pressure-threshold-)))))))
  (:arm-potentio-vector
    (arm)
    (case arm
      (:larm (subseq (send *ri* :state :potentio-vector) 1 9))
      (:rarm (subseq (send *ri* :state :potentio-vector) 9 17))
      (t nil)))
  (:tf-pose->coords
    (frame_id pose)
    (let ((base-name (send (send robot :base_lk) :name)))
      (send (send *tfl* :lookup-transform base-name frame_id (ros::time 0))
            :transform (ros::tf-pose->coords pose))))
  (:wait-interpolation-until-grasp
    (arm)
    (when (send self :simulation-modep)
      (return-from :wait-interpolation-until-grasp (send self :wait-interpolation))
      )
    (while (send self :interpolatingp)
           (if (send self :graspingp arm)
             (progn
               (ros::ros-info "[:wait-interpolation-until-grasp] Grasping detected. Cancel angle vector: ~a" arm)
               (send self :cancel-angle-vector)))))
  (:calib-pressure-threshold
    (&optional (arm :arms))
    (send self :start-grasp arm)
    (dolist (l/r (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (let ((min-pressure)
            (topic (format nil "/gripper_front/limb/~a/pressure/state" (arm-to-str l/r))))
        (dotimes (i 7)
          (let ((pressure (send (one-shot-subscribe topic std_msgs::Float64) :data)))
            (when (or (null min-pressure) (< pressure min-pressure)) (setq min-pressure pressure)))
          (unix::sleep 1))
        (cond ((eq l/r :larm)
               (setq larm-pressure-threshold- (- min-pressure 5)))
              ((eq l/r :rarm)
               (setq rarm-pressure-threshold- (- min-pressure 10))))))
    (send self :stop-grasp arm)
    (ros::ros-info "[:calib-pressure-threshold] Threshold r: ~a l: ~a"
                   rarm-pressure-threshold- larm-pressure-threshold-)))
