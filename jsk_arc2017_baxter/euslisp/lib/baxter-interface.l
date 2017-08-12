;; -*- mode: lisp;-*-

(require "package://jsk_arc2017_baxter/euslisp/lib/baxter.l")
(require "package://baxtereus/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")


(ros::load-ros-manifest "jsk_arc2017_baxter")

(unless (find-package "JSK_ARC2017_BAXTER")
  (make-package "JSK_ARC2017_BAXTER"))

(defclass jsk_arc2017_baxter::baxter-interface
  :super baxter-interface
  :slots (rarm-pressure-threshold-
          larm-pressure-threshold-
          proximity-threshold-
          hand-actions-
          prismatic-calib-actions-
          pressures-
          finger-flex-
          finger-load-
          prismatic-load-
          prismatic-vel-
          proximities-
          proximity-init-values-))

(defmethod jsk_arc2017_baxter::baxter-interface
  (:init
    (&rest args &key
           ((:moveit-environment mvit-env)
            (instance jsk_arc2017_baxter::baxter-moveit-environment))
           ((:moveit-robot mvit-rb) (instance jsk_arc2017_baxter::baxter-robot :init))
           &allow-other-keys)
    ;; initialize controllers
    (send-super* :init :joint-states-queue-size 3
                 ;; FIXME: :robot jsk_arc2017_baxter::baxter-robot
                 ;; FIXME: :end-coords-interpolation does not work.
                 :robot baxter-robot
                 :moveit-environment nil
                 :moveit-robot nil args)
    (send self :add-controller :rarm-head-controller)
    (send self :add-controller :larm-head-controller)
    (send self :add-controller :no-gripper-controller)
    (send self :add-controller :rarm-no-gripper-controller)
    (send self :add-controller :larm-no-gripper-controller)
    (send self :add-controller :rarm-head-no-gripper-controller)
    (send self :add-controller :larm-head-no-gripper-controller)
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
    (setq hand-actions- (make-hash-table))
    (setq prismatic-calib-actions- (make-hash-table))
    (dolist (arm (list :rarm :larm))
      ;; initialize hand action
      (sethash arm hand-actions-
               (instance ros::simple-action-client :init
                         (format nil "/gripper_front/limb/~a/hand/follow_joint_trajectory"
                                 (arm2str arm))
                         control_msgs::FollowJointTrajectoryAction
                         :groupname groupname))
      ;; check if hand action is respond (based on baxter-interface)
      (unless
        (and joint-action-enable (send (gethash arm hand-actions-) :wait-for-server 3))
        (ros::ros-warn "~A is not respond" (gethash arm hand-actions-))
        (ros::ros-info "*** if you do not have hand, you can ignore this message ***"))
      ;; initialize action for prismatic joint calib
      (sethash arm prismatic-calib-actions-
               (instance ros::simple-action-client :init
                         (format nil "/gripper_front/limb/~a/dxl/prismatic_joint_controller/calib"
                                 (arm2str arm))
                         jsk_arc2017_baxter::CalibJointAction
                         :groupname groupname))
      ;; check if action is respond
      (unless
        (and joint-action-enable (send (gethash arm prismatic-calib-actions-) :wait-for-server 3))
        (ros::ros-error "~A is not respond" (gethash arm prismatic-calib-actions-))
        (ros::ros-error "You cannot calibrate gripper joint")))
    ;; gripper sensors
    (setq pressures- (make-hash-table))
    (setq finger-load- (make-hash-table))
    (setq finger-flex- (make-hash-table))
    (setq prismatic-load- (make-hash-table))
    (setq prismatic-vel- (make-hash-table))
    (setq proximities- (make-hash-table))
    (setq proximity-init-values- (make-hash-table))
    (dolist (arm (list :rarm :larm))
      (ros::subscribe (format nil "/gripper_front/limb/~a/pressure/state" (arm2str arm))
                      std_msgs::Float64
                      #'send self :set-pressure arm
                      10
                      :groupname groupname)
      (ros::subscribe (format nil "/gripper_front/limb/~a/dxl/finger_tendon_controller/state"
                              (arm2str arm))
                      dynamixel_msgs::JointState
                      #'send self :set-finger-load arm
                      :groupname groupname)
      (ros::subscribe (format nil "/gripper_front/limb/~a/dxl/prismatic_joint_controller/state"
                              (arm2str arm))
                      dynamixel_msgs::JointState
                      #'send self :set-prismatic-state arm
                      :groupname groupname)
      (ros::subscribe (format nil "/gripper_front/limb/~a/proximity_array"
                              (arm2str arm))
                      force_proximity_ros::ProximityArray
                      #'send self :set-proximities arm
                      :groupname groupname)
      (sethash arm finger-flex- (make-hash-table))
      (sethash arm proximity-init-values- (make-hash-table))
      (sethash arm proximities- (make-hash-table))
      (dolist (side (list :right :left))
        (ros::subscribe (format nil "/gripper_front/limb/~a/flex/~a/state"
                                (arm2str arm) (symbol2str side))
                        std_msgs::UInt16
                        #'send self :set-finger-flex arm side
                        :groupname groupname)))
    (if mvit-rb (setq moveit-robot mvit-rb))
    (if mvit-env (send self :set-moveit-environment (send mvit-env :init :robot moveit-robot))))
  ;; Overwrite super class's :rarm-controller
  (:rarm-controller ()
    (append
      (send-super :rarm-controller)
      (send self :rgripper-controller)))
  ;; Overwrite super class's :larm-controller
  (:larm-controller ()
    (append
      (send-super :larm-controller)
      (send self :lgripper-controller)))
  (:rarm-head-controller ()
    (append
      (send-super :head-controller)
      (send self :rarm-controller)))
  (:larm-head-controller ()
    (append
      (send-super :head-controller)
      (send self :larm-controller)))
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
  (:rarm-head-no-gripper-controller ()
    (append
      (send-super :head-controller)
      (send self :rarm-no-gripper-controller)))
  (:larm-head-no-gripper-controller ()
    (append
      (send-super :head-controller)
      (send self :larm-no-gripper-controller)))
  (:no-gripper-controller ()
    (append
      (send self :rarm-no-gripper-controller)
      (send self :larm-no-gripper-controller)))
  (:rgripper-controller ()
    (list
      (list
        (cons :controller-action "/gripper_front/limb/right/vacuum_gripper/follow_joint_trajectory")
        (cons :controller-state "/gripper_front/limb/right/vacuum_gripper/state")
        (cons :action-type control_msgs::FollowJointTrajectoryAction)
        (cons :joint-names (list "right_gripper_prismatic_joint" "right_gripper_vacuum_pad_joint")))))
  (:lgripper-controller ()
    (list
      (list
        (cons :controller-action "/gripper_front/limb/left/vacuum_gripper/follow_joint_trajectory")
        (cons :controller-state "/gripper_front/limb/left/vacuum_gripper/state")
        (cons :action-type control_msgs::FollowJointTrajectoryAction)
        (cons :joint-names (list "left_gripper_prismatic_joint" "left_gripper_vacuum_pad_joint")))))
  (:set-pressure
    (arm msg)
    (let ((ps (gethash arm pressures-)))
      (pushback (send msg :data) ps)
      (while (> (length ps) 10)
        (pop ps))
      (sethash arm pressures- ps)))
  (:set-finger-load
    (arm msg)
    (sethash arm finger-load- (send msg :load)))
  (:set-prismatic-state
    (arm msg)
    (sethash arm prismatic-load- (send msg :load))
    (sethash arm prismatic-vel- (send msg :velocity)))
  (:set-finger-flex
    (arm side msg)
    (sethash side (gethash arm finger-flex-) (send msg :data)))
  (:set-proximities
    (arm msg)
    (sethash :left (gethash arm proximities-) (send (elt (send msg :proximities) 0) :average))
    (sethash :right (gethash arm proximities-) (send (elt (send msg :proximities) 1) :average)))

  ;; Hand interface
  ;; based on naoqi-interface and fetch-interface
  (:move-hand
    (arm av tm &key (wait t))
    (if (and (= (length av) 2) (not (send self :simulation-modep)))
      (let (av-list res (goal (instance control_msgs::FollowJointTrajectoryActionGoal :init))
                    (traj (instance trajectory_msgs::JointTrajectory :init)))
        (dotimes (i (length av)) (pushback (elt av i) av-list))
        (send traj :header :seq 1)
        (send traj :header :stamp (ros::time-now))
        (send traj :joint_names
              (list (format nil "~a_gripper_finger_yaw_joint" (arm2str arm))
                    (format nil "~a_gripper_finger_roll_joint" (arm2str arm))))
        (send traj :points
              (list (instance trajectory_msgs::JointTrajectoryPoint
                              :init
                              :positions (mapcar #'deg2rad av-list)
                              :time_from_start (ros::time (/ tm 1000)))))
        (send goal :goal :trajectory traj)
        (send (gethash arm hand-actions-) :send-goal goal)
        (setq res (send (gethash arm hand-actions-) :get-result))
        (when wait
          (send (gethash arm hand-actions-) :wait-for-result)
          (setq res (send (gethash arm hand-actions-) :get-result)))
        (unless (eq (send res :error_code) 0)
          (warning-message 3 ":move-hand error code ~A returns from JTA: ~A~%"
                           (send res :error_code) (send res :error_string)))
        res)
      nil))
  (:cancel-move-hand (arm)
    (send (gethash arm hand-actions-) :cancel-goal))
  (:hand-interpolatingp (arm)
    (send self :spin-once)
    (eq (send (gethash arm hand-actions-) :get-state) ros::*simple-goal-state-active*))
  (:get-finger-flex (arm side)
    (send self :spin-once)
    (gethash side (gethash arm finger-flex-)))
  (:get-finger-load (arm)
    (send self :spin-once)
    (gethash arm finger-load-))
  (:get-prismatic-load (arm)
    (send self :spin-once)
    (gethash arm prismatic-load-))
  (:get-prismatic-vel (arm)
    (send self :spin-once)
    (gethash arm prismatic-vel-))
  (:get-proximity (arm side &key (raw nil))
    (send self :spin-once)
    (let ((l-init (gethash :left (gethash arm proximity-init-values-)))
          (r-init (gethash :right (gethash arm proximity-init-values-)))
          (l (gethash :left (gethash arm proximities-)))
          (r (gethash :right (gethash arm proximities-))))
      (if (null raw)
        (cond
          ;; for sparkfun proximity sensor
          ((eq side :left)
           (/ (- l l-init) (expt (/ l-init 2500.0) 1.5)))
          ((eq side :right)
           (/ (- r r-init) (expt (/ r-init 2500.0) 1.5))))
        (cond
          ((eq side :left) l)
          ((eq side :right) r)))))
  (:get-real-finger-av (arm)
    (send self :update-robot-state :wait-until-update t)
    (float-vector
      (send robot (str2symbol (format nil "~a_gripper_finger_yaw_joint" (arm2str arm)))
            :joint-angle)
      (send robot (str2symbol (format nil "~a_gripper_finger_roll_joint" (arm2str arm)))
            :joint-angle)))
  (:finger-closep (arm)
    (> (aref (send self :get-real-finger-av arm) 1) 30))
  (:start-prismatic-calib (arm)
    (send (gethash arm prismatic-calib-actions-)
          :send-goal (instance jsk_arc2017_baxter::CalibJointActionGoal :init)))
  (:prismatic-calibratingp (arm)
    (send self :spin-once)
    (eq (send (gethash arm prismatic-calib-actions-) :get-state) ros::*simple-goal-state-active*))

  (:get-arm-controller (arm &key (gripper t) (head nil))
    (cond ((eq arm :rarm)
           (cond ((and head gripper) :rarm-head-controller)
                 ((and (null head) gripper) :rarm-controller)
                 ((and head (null gripper)) :rarm-head-no-gripper-controller)
                 ((and (null head) (null gripper)) :rarm-no-gripper-controller)))
          ((eq arm :larm)
           (cond ((and head gripper) :larm-head-controller)
                 ((and (null head) gripper) :larm-controller)
                 ((and head (null gripper)) :larm-head-no-gripper-controller)
                 ((and (null head) (null gripper)) :larm-no-gripper-controller)))
          (t nil)))
  (:start-grasp
    (&optional (arm :arms) (type :suction))
    (dolist (l/r (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (cond ((eq type :suction)
             (if (ros::get-param "/apc_on_gazebo" nil)
               (ros::service-call
                 (format nil "/robot/~a_vacuum_gripper/on" (arm2str l/r))
                 (instance std_srvs::EmptyRequest :init))
               (ros::publish
                 (format nil "/vacuum_gripper/limb/~a" (arm2str l/r))
                 (instance std_msgs::Bool :init :data t))))
            ((eq type :pinch)
             (send self :update-robot-state :wait-until-update t)
             (let ((finger-av (send self :get-real-finger-av l/r))
                   prev-av av avs hand-interpolatingp)
               ;; rotate arm using proximity sensor
               (setf (aref finger-av 1) 180)
               (send self :move-hand l/r finger-av 2000 :wait nil)
               (dotimes (x 100)
                 (if (send self :interpolatingp) (return))
                 (unix::usleep 1000))
               (while (and
                        (< (max (send self :get-proximity l/r :right)
                                (send self :get-proximity l/r :left))
                           proximity-threshold-)
                        (setq hand-interpolatingp (send self :hand-interpolatingp l/r)))
                 (unix::usleep 1000))
               (when hand-interpolatingp
                 (send self :cancel-move-hand l/r)
                 (send self :update-robot-state :wait-until-update t)
                 (setq prev-av (send robot :angle-vector))
                 (if (< (send self :get-proximity l/r :right) (send self :get-proximity l/r :left))
                   (progn
                     (send robot l/r :wrist-r :joint-angle 45 :relative t)
                     (setq av (send robot :angle-vector))
                     (send robot :angle-vector prev-av)
                     (send self :angle-vector-raw av 2000)
                     (send self :move-hand l/r finger-av 4000 :wait nil)
                     (dotimes (x 100)
                       (if (send self :interpolatingp) (return))
                       (unix::usleep 1000))
                     (while (and (< (send self :get-proximity l/r :right)
                                    (send self :get-proximity l/r :left))
                                 (send self :interpolatingp))
                       (unix::usleep 1000)))
                   (progn
                     (send robot l/r :wrist-r :joint-angle -45 :relative t)
                     (setq av (send robot :angle-vector))
                     (send robot :angle-vector prev-av)
                     (send self :angle-vector-raw av 2000)
                     (send self :move-hand l/r finger-av 4000 :wait nil)
                     (dotimes (x 100)
                       (if (send self :interpolatingp) (return))
                       (unix::usleep 1000))
                     (while (and (> (send self :get-proximity l/r :right)
                                    (send self :get-proximity l/r :left))
                                 (send self :interpolatingp))
                       (unix::usleep 1000))))
                 (send self :cancel-move-hand l/r)
                 (send self :cancel-angle-vector
                       :controller-type (send self :get-arm-controller arm))
                 (send self :update-robot-state :wait-until-update t)
                 (send self :move-hand l/r finger-av 1000))
               (setq prev-av (send robot :angle-vector))
               (when (> (aref finger-av 0) 45)
                 ;; if cylindrical and spherical grasp, move other gripper joints
                 (send robot l/r :gripper-p :joint-angle -90)
                 (pushback (send robot :angle-vector) avs)
                 ;(send robot l/r :gripper-x :joint-angle 0)
                 ;(pushback (send robot :angle-vector) avs)
                 (send robot :angle-vector prev-av)
                 (send self :angle-vector-sequence-raw avs)
                 (send self :wait-interpolation)))))))
  (:stop-grasp
    (&optional (arm :arms) (type :suction))
    (dolist (l/r (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (cond ((eq type :suction)
             (if (ros::get-param "/apc_on_gazebo" nil)
               (ros::service-call
                 (format nil "/robot/~a_vacuum_gripper/off" (arm2str l/r))
                 (instance std_srvs::EmptyRequest :init))
               (ros::publish
                 (format nil "/vacuum_gripper/limb/~a" (arm2str l/r))
                 (instance std_msgs::Bool :init :data nil))))
            ((eq type :pinch)
             (send self :update-robot-state :wait-until-update t)
             (let ((finger-av (send self :get-real-finger-av l/r))
                   (prev-av (send robot :angle-vector)) avs)
               (setf (aref finger-av 1) 0)
               (send self :move-hand l/r finger-av 1000 :wait nil)
               (if (> (aref finger-av 0) 45)
                 (progn
                   ;; if cylindrical and spherical grasp, move other gripper joints
                   (pushback (send robot :angle-vector) avs)
                   (send robot l/r :gripper-p :joint-angle 0)
                   (pushback (send robot :angle-vector) avs)
                   ;(send robot l/r :gripper-x :joint-angle 0)
                   ;(pushback (send robot :angle-vector) avs)
                   (send robot :angle-vector prev-av)
                   (send self :angle-vector-sequence-raw avs)
                   (send self :wait-interpolation))
                 (unix::sleep 1)))))))
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
    (arm &optional (type :suction))
    (let (topic finger-av p-thre pressures suctioningp pinchingp)
      (if (ros::get-param "/apc_on_gazebo" nil)
        (progn
          (setq topic (format nil "/robot/~a_vacuum_gripper/grasping" (arm-to-str arm)))
          (send (one-shot-subscribe topic std_msgs::Bool) :data)
          )
        (progn
          (setq finger-av (send self :get-real-finger-av arm))
          (setq p-thre
                (cond
                  ((eq arm :rarm) rarm-pressure-threshold-)
                  ((eq arm :larm) larm-pressure-threshold-)))
          (send self :spin-once)
          (setq pressures (gethash arm pressures-))
          (setq suctioningp
                (< (car (reverse pressures)) p-thre))
          (unless suctioningp
            (let ((tcount 0))
              (dolist (p pressures)
                (when (< p p-thre) (incf tcount)))
              (when (> tcount (/ (length pressures) 2)) (setq suctioningp t))))
          ;; FIXME: cannot detect grasping small objects
          ;; (setq pinchingp
          ;;       (and (or
          ;;              (and (< (aref finger-av 0) 45) (< (aref finger-av 1) 80))
          ;;              (and (>= (aref finger-av 0) 45) (< (aref finger-av 1) 155)))
          ;;            (< (send self :get-finger-load arm) -0.55)))
          (setq pinchingp t)
          (cond ((eq type :suction) suctioningp)
                ((eq type :pinch) pinchingp)
                ((null type) (or suctioningp pinchingp))
                (t (ros::ros-error "Unsupported type for :graspingp") (exit)))))))
  (:arm-potentio-vector
    (arm)
    (case arm
      (:larm (subseq (send self :state :potentio-vector) 1 10))
      (:rarm (subseq (send self :state :potentio-vector) 10 19))
      (t nil)))
  (:tf-pose->coords
    (frame_id pose)
    (let ((base-name (send (send robot :base_lk) :name)))
      (send (send *tfl* :lookup-transform base-name frame_id (ros::time 0))
            :transform (ros::tf-pose->coords pose))))
  (:wait-interpolation-until-grasp
    (arm)
    (when (send self :simulation-modep)
      (return-from :wait-interpolation-until-grasp (send self :wait-interpolation)))
    ;; wait for :interpolatingp
    (unix::usleep 1000)
    (while (send self :interpolatingp)
      (if (send self :graspingp arm)
        (progn
          (ros::ros-info "[:wait-interpolation-until-grasp] Grasping detected. Cancel angle vector: ~a" arm)
          (send self :cancel-angle-vector
                :controller-type (send self :get-arm-controller arm))))
      (unix::usleep 1000)))
  (:wait-interpolation-until
    (arm &rest args)
    (when (send self :simulation-modep)
      (return-from :wait-interpolation-until (send self :wait-interpolation)))
    (unless (or (eq arm :rarm) (eq arm :larm))
      (error ":wait-interpolation-until set arm for first arg~%"))
    (let (conds r-init-flex l-init-flex init-load r-flex l-flex
                finger-load prismatic-load flex-thre-offset
                r-proximity l-proximity)
      (setq r-init-flex (send self :get-finger-flex arm :right))
      (setq l-init-flex (send self :get-finger-flex arm :left))
      (setq init-load (send self :get-finger-load arm))
      (setq flex-thre-offset
            (if (member :flex-thre-offset args)
              (cadr (member :flex-thre-offset args)) nil))
      (ros::ros-info
        "[:wait-interpolation-until] Init flex:  r: ~a l: ~a" r-init-flex l-init-flex)
      (ros::ros-info "[:wait-interpolation-until] Init load: ~a" init-load)
      (when (member :grasp args)
        (ros::ros-info "[:wait-interpolation-until] Prepare for grasp")
        (pushback
          #'(lambda ()
              (if (send self :graspingp arm)
                (progn
                  (ros::ros-info "[:wait-interpolation-until] Detected grasping")
                  t)
                nil))
          conds))
      (when (member :ungrasp args)
        (ros::ros-info "[:wait-interpolation-until] Prepare for ungrasp")
        (pushback
          #'(lambda ()
              (if (not (send self :graspingp arm))
                (progn
                  (ros::ros-info "[:wait-interpolation-until] Detected ungrasping")
                  t)
                nil))
          conds))
      (when (member :finger-flexion args)
        (ros::ros-info "[:wait-interpolation-until] Prepare for finger flexion")
        (pushback
          #'(lambda ()
              (setq r-flex (send self :get-finger-flex arm :right))
              (setq l-flex (send self :get-finger-flex arm :left))
              (if (or (> r-flex (+ r-init-flex (if flex-thre-offset flex-thre-offset 20)))
                      (> l-flex (+ l-init-flex (if flex-thre-offset flex-thre-offset 20))))
                (progn
                  (ros::ros-info "[:wait-interpolation-until] Detected finger flexion: r: ~a l: ~a"
                                 r-flex l-flex)
                  t)
                nil))
          conds))
      (when (member :finger-extension args)
        (ros::ros-info "[:wait-interpolation-until] Prepare for finger extension")
        (pushback
          #'(lambda ()
              (setq r-flex (send self :get-finger-flex arm :right))
              (setq l-flex (send self :get-finger-flex arm :left))
              (if (or (< r-flex (- r-init-flex (if flex-thre-offset flex-thre-offset 30)))
                      (< l-flex (- l-init-flex (if flex-thre-offset flex-thre-offset 30))))
                (progn
                  (ros::ros-info "[:wait-interpolation-until] Detected finger extension r: ~a l: ~a"
                                 r-flex l-flex)
                  t)
                nil))
          conds))
      (when (member :finger-loaded args)
        (ros::ros-info "[:wait-interpolation-until] Prepare for finger loaded")
        (pushback
          #'(lambda ()
              (setq finger-load (send self :get-finger-load arm))
              (if (< finger-load (- init-load 0.01))
                (progn
                  (ros::ros-info "[:wait-interpolation-until] Detected finger loaded load: ~a"
                                 finger-load)
                  t)
                nil))
          conds))
      (when (member :finger-unloaded args)
        (ros::ros-info "[:wait-interpolation-until] Prepare for finger unloaded")
        (pushback
          #'(lambda ()
              (setq finger-load (send self :get-finger-load arm))
              (if (> finger-load (+ init-load 0.01))
                (progn
                  (ros::ros-info "[:wait-interpolation-until] Detected finger unloaded load: ~a"
                                 finger-load)
                  t)
                nil))
          conds))
      (when (member :prismatic-loaded args)
        (ros::ros-info "[:wait-interpolation-until] Prepare for prismatic joint loaded")
        (pushback
          #'(lambda ()
              (setq prismatic-load (send self :get-prismatic-load arm))
              (if (and (< prismatic-load -0.07) (< (send self :get-prismatic-vel arm) 0.01))
                (progn
                  (ros::ros-info "[:wait-interpolation-until] Detected prismatic loaded load: ~a"
                                 prismatic-load)
                  t)
                nil))
          conds))
      (when (member :finger-proximity args)
        (ros::ros-info "[:wait-interpolation-until] Prepare for finger proximity")
        (pushback
          #'(lambda ()
              (setq r-proximity (send self :get-proximity arm :right))
              (setq l-proximity (send self :get-proximity arm :left))
              (if (> (max r-proximity l-proximity) proximity-threshold-)
                (progn
                  (ros::ros-info "[:wait-interpolation-until] Detected finger proximity. r: ~a, l: ~a"
                                 r-proximity l-proximity)
                  t)
                nil))
          conds))
      ;; wait for :interpolatingp
      (dotimes (x 100)
        (if (send self :interpolatingp) (return))
        (unix::usleep 1000))
      (while (send self :interpolatingp)
        (when (reduce #'(lambda (x y) (or x y)) (mapcar #'funcall conds))
          (ros::ros-info "[:wait-interpolation-until] Cancel angle vector: ~a" arm)
          (send self :cancel-angle-vector
                :controller-type (send self :get-arm-controller arm)))
        (unix::usleep 1000))))
  (:calib-pressure-threshold
    (&optional (arm :arms))
    (send self :start-grasp arm)
    (dolist (l/r (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (let (pressure min-pressure)
        (dotimes (i 7)
          (send self :spin-once)
          (setq pressure (car (reverse (gethash l/r pressures-))))
          (when (or (null min-pressure) (< pressure min-pressure))
            (setq min-pressure pressure))
          (unix::sleep 1))
        (cond ((eq l/r :larm)
               (setq larm-pressure-threshold- (- min-pressure 15)))
              ((eq l/r :rarm)
               (setq rarm-pressure-threshold- (- min-pressure 15))))))
    (send self :stop-grasp arm)
    (ros::ros-info "[:calib-pressure-threshold] Threshold r: ~a l: ~a"
                   rarm-pressure-threshold- larm-pressure-threshold-))
  (:calib-proximity-threshold
    (&optional (arm :arms))
    (send self :spin-once)
    (dolist (l/r (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (sethash :left (gethash l/r proximity-init-values-)
               (send self :get-proximity l/r :left :raw t))
      (sethash :right (gethash l/r proximity-init-values-)
               (send self :get-proximity l/r :right :raw t))
      ;; TODO: decide proper threshold 100 -> ???
      (setq proximity-threshold- 100)))
  (:speak
    (msg &key (wait nil))
    (assert (stringp msg) (format nil "Invalid msg, though string is expected: ~a" msg))
    (speak-en msg :wait wait)
    )
  )

(defclass jsk_arc2017_baxter::baxter-moveit-environment
  :super moveit-environment)
(defmethod jsk_arc2017_baxter::baxter-moveit-environment
  (:init (&key ((:robot rb) (instance jsk_arc2017_baxter::baxter-robot :init)) &rest args)
         (send-super* :init :robot rb :frame-id "world" args))
  (:default-configuration ()
   (list (list :rarm
               (cons :group-name "right_arm")
               (cons :target-link
                     (send self :search-link-from-name "right_gripper_pad_with_base"))
               (cons :joint-list (send robot :rarm :joint-list))
               )
         (list :larm
               (cons :group-name "left_arm")
               (cons :target-link
                     (send self :search-link-from-name "left_gripper_pad_with_base"))
               (cons :joint-list (send robot :larm :joint-list)))
         (list :arms
               (cons :group-name "both_arms")
               (cons :target-link
                     (list
                       (send self :search-link-from-name "left_gripper_pad_with_base")
                       (send self :search-link-from-name "right_gripper_pad_with_base")))
               (cons :joint-list (append (send robot :larm :joint-list)
                                         (send robot :rarm :joint-list)))))))

(defun jsk_arc2017_baxter::baxter-init
  (&key (type :default-controller) (moveit nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance jsk_arc2017_baxter::baxter-moveit-environment))
      (setq mvit-rb (instance jsk_arc2017_baxter::baxter-robot :init)))
    (when (not (boundp '*ri*))
      (setq *ri* (instance jsk_arc2017_baxter::baxter-interface :init
                           :type type
                           :moveit-environment mvit-env
                           :moveit-robot mvit-rb)))
    (when (not (boundp '*baxter*))
      (setq *baxter* (instance jsk_arc2017_baxter::baxter-robot :init))
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector)))))
