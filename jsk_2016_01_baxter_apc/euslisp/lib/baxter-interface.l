;; -*- mode: lisp;-*-

(require "package://baxtereus/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")

(ros::load-ros-manifest "jsk_2016_01_baxter_apc")

(unless (find-package "JSK_2016_01_BAXTER_APC")
  (make-package "JSK_2016_01_BAXTER_APC"))

(defclass jsk_2016_01_baxter_apc::baxter-interface
  :super baxter-interface
  :slots (tfl-
          bin-boxes-
          objects-in-bin-boxes-
          objects-in-bin-coms-
          objects-sib-boxes-
          objects-sib-coords-
          bin-coords-list-
          hard-coded-variables-
          last-av-before-approach-
          order-bin-box-
          order-bin-movable-region-
          objects-in-order-bin-boxes-
          objects-in-order-bin-coords-))

(defmethod jsk_2016_01_baxter_apc::baxter-interface
  (:init
    (&rest args &key
           ((:moveit-environment mvit-env)
            (instance jsk_2016_01_baxter_apc::baxter-moveit-environment))
           ((:moveit-robot mvit-rb) (instance jsk_2016_01_baxter_apc::baxter-robot :init))
           &allow-other-keys)
    ;; initialize controllers
    (send-super* :init :joint-states-queue-size 2
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
    (setq tfl- (instance ros::transform-listener :init))
    (setq bin-boxes- (make-hash-table))
    (setq objects-in-bin-boxes- (make-hash-table))
    (setq objects-in-bin-coms- (make-hash-table))
    (setq objects-sib-boxes- (make-hash-table))
    (setq objects-sib-coords- (make-hash-table))
    ;; set hard-coded variables
    (setq hard-coded-variables- (make-hash-table))
    (sethash :r-pressure-threshold hard-coded-variables- 1000)
    (sethash :l-pressure-threshold hard-coded-variables- 1000)
    (sethash :offset-avoid-bin-top hard-coded-variables- -20)
    (sethash :pad-link-l hard-coded-variables- 85)
    (sethash :gripper-tube-t hard-coded-variables- 40)
    (sethash :offset-object-h hard-coded-variables- -10)
    (sethash :offset-object-w hard-coded-variables- 20)
    (sethash :offset-gripper-bin-top hard-coded-variables- 10)
    (sethash :offset-gripper-bin-side hard-coded-variables- 40)
    (sethash :draw-out-from-bin-l hard-coded-variables- 100)
    (sethash :default-overlook-angle hard-coded-variables- (/ pi 4.5))
    (sethash :default-overlook-offset hard-coded-variables- (make-hash-table))
    (dolist (bin (list :a :b :c :d :e :f :g :h :i :j :k :l))
      (sethash bin (gethash :default-overlook-offset hard-coded-variables-)
               (if (find bin (list :b :i :k))
                 (float-vector -280 0 -60)
                 (float-vector -280 0 -10)
                 )
               )
      )
    (sethash :bin-reachable-depth-gripper-90 hard-coded-variables- 200)
    (setq order-bin-box- (make-hash-table))
    (setq order-bin-movable-region- (make-hash-table))
    (setq objects-in-order-bin-boxes- (make-hash-table))
    (setq objects-in-order-bin-coords- (make-hash-table))
    (unless (ros::get-param "/apc_on_gazebo" nil)
      (ros::advertise "/vacuum_gripper/limb/left" std_msgs::Bool)
      (ros::advertise "/vacuum_gripper/limb/right" std_msgs::Bool)
      )
    (ros::advertise "/gripper_front/limb/left/servo/torque" std_msgs::Bool)
    (ros::advertise "/gripper_front/limb/right/servo/torque" std_msgs::Bool)
    (if mvit-rb (setq moveit-robot mvit-rb))
    (if mvit-env (send self :set-moveit-environment (send mvit-env :init :robot moveit-robot)))
    )

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

  (:start-grasp
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (send self :arm-symbol2str arm))))
      (if (ros::get-param "/apc_on_gazebo" nil)
        (ros::service-call
          (format nil "/robot/~a_vacuum_gripper/on" l/r)
          (instance std_srvs::EmptyRequest :init))
        (ros::publish
          (format nil "/vacuum_gripper/limb/~a" l/r)
          (instance std_msgs::Bool :init :data t))
        )
      ))
  (:stop-grasp
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (send self :arm-symbol2str arm))))
      (if (ros::get-param "/apc_on_gazebo" nil)
        (ros::service-call
          (format nil "/robot/~a_vacuum_gripper/off" l/r)
          (instance std_srvs::EmptyRequest :init))
        (ros::publish
          (format nil "/vacuum_gripper/limb/~a" l/r)
          (instance std_msgs::Bool :init :data nil))
        )
      ))
  (:gripper-servo-on
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (send self :arm-symbol2str arm))))
      (ros::publish
        (format nil "/gripper_front/limb/~a/servo/torque" l/r)
        (instance std_msgs::Bool :init :data t))
      ))
  (:gripper-servo-off
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (send self :arm-symbol2str arm))))
      (ros::publish
        (format nil "/gripper_front/limb/~a/servo/torque" l/r)
        (instance std_msgs::Bool :init :data nil))
      ))
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
               ((eq arm :rarm)
                (gethash :r-pressure-threshold hard-coded-variables-))
               ((eq arm :larm)
                (gethash :l-pressure-threshold hard-coded-variables-))
               )
             )
          )
        )
      )
    )
  (:opposite-arm (arm) (if (eq arm :larm) :rarm :larm))
  (:need-to-wait-opposite-arm
    (arm)
    (let (opposite-state opposite-target)
      (setq opposite-state
            (str2symbol (ros::get-param
                          (format nil "~a_hand/state"
                                  (send self :arm-symbol2str
                                        (send self :opposite-arm arm))))))
      (setq opposite-target
            (str2symbol (ros::get-param
                          (format nil "~a_hand/target_bin"
                                  (send self :arm-symbol2str
                                        (send self :opposite-arm arm))))))
      (if (and (eq opposite-state :pick_object) (find opposite-target (list :b :e :h :k)))
        t
        nil)))
  (:keep-picking-p
    (bin past-graspingps object-found-p &key (max-retry-grasp-num 3))
    (let ((latest-grasp-failure-num 0))
      (dolist (p (reverse past-graspingps))
        (if p
          (return)
          (1+ latest-grasp-failure-num))
        )
      (if (< latest-grasp-failure-num max-retry-grasp-num)
        (and (> (+ (length (send self :get-bin-contents bin)) 1)
                (length past-graspingps)) object-found-p)
        nil)
      )
    )
  (:bin-reachable-depth
    (gripper-angle)
    (+ (gethash :bin-reachable-depth-gripper-90 hard-coded-variables-)
       (* (gethash :pad-link-l hard-coded-variables-) (cos (deg2rad gripper-angle))))
    )
  (:set-object-segmentation-candidates
    (&key arm candidates)
    (let ((req (instance jsk_recognition_msgs::SetLabelsRequest :init)))
      (send req :labels candidates)
      (ros::service-call
        (format nil "/~a_hand_camera/apply_context_to_label_proba/update_candidates"
                (arm2str arm))
        req)
      )
    )
  (:need-to-wait-opposite-arm-for-stow
    (arm &key (state))
    (let (opposite-state opposite-target arm-target)
      (setq opposite-state
            (str2symbol (ros::get-param
                          (format nil "~a_hand/state"
                                  (send self :arm-symbol2str
                                        (send self :opposite-arm arm))))))
      (setq opposite-target
            (str2symbol (ros::get-param
                          (format nil "~a_hand/target_bin"
                                  (send self :arm-symbol2str
                                        (send self :opposite-arm arm))))))
      (setq arm-target
            (str2symbol (ros::get-param
                          (format nil "~a_hand/target_bin"
                                  (send self :arm-symbol2str arm)))))
      (cond
        ((or (eq state :pick_object) (eq state :wait_for_opposite_arm_in_pick))
         (or (eq opposite-state :pick_object)
             (and (or (eq opposite-state :place_object)
                      (eq opposite-state :return_from_bin))
                  (find opposite-target (list :b :e :h :k)))))
        ((or (eq state :place_object) (eq state :wait_for_opposite_arm_in_place))
         (or (and (eq opposite-state :place_object)
                  (find opposite-target (list :b :e :h :k)))
             (and (eq opposite-state :return_from_bin)
                  (or (find opposite-target (list :b :e :h :k))
                      (find arm-target (list :b :e :h :k))))))
        ((or (eq state :return_from_bin) (eq state :wait_for_opposite_arm_in_return))
         (and (eq opposite-state :pick_object)
              (not (find opposite-target (list :b :e :h :k)))))
        ((eq state :wait_opposite_arm_start_picking)
         (null (eq opposite-state :pick_object)))
        (t (and (eq opposite-state :pick_object) (find opposite-target (list :b :e :h :k)))))))
  (:arm-symbol2str
    (arm)
    (case arm
      (:larm "left")
      (:rarm "right")))
  (:arm-potentio-vector
    (arm)
    (case arm
      (:larm (subseq (send *ri* :state :potentio-vector) 1 9))
      (:rarm (subseq (send *ri* :state :potentio-vector) 9 17))
      (t nil)
      )
    )
  (:tf-pose->coords
    (frame_id pose)
    (let (coords)
      (setq coords (ros::tf-pose->coords pose))
      (send (send tfl- :lookup-transform "base" frame_id (ros::time 0)) :transform coords)
      ))
  (:fold-pose-back
    (&optional (arm :arms))
    (dolist (tmp-arm (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (send *baxter* :fold-pose-back tmp-arm))
    (send *ri* :send-av))
  (:detect-target-object-in-bin
    (target-object bin)
    (let (obj-sizes box-sizes obj-idx box-idx box-size box-sizes-back
                    (candidates (send self :get-bin-contents bin))
                    (boxes (gethash bin objects-in-bin-boxes-)))
      (dolist (obj candidates) (pushback (get-object-size obj) obj-sizes))
      (dolist (box boxes) (pushback (send (send self :bbox->cube box) :volume) box-sizes))
      (setq box-sizes-bak (copy-seq box-sizes))
      (while (and obj-sizes box-sizes)
             (setq obj-idx (argmax obj-sizes))
             (setq obj-sizes (remove (elt obj-sizes obj-idx) obj-sizes))
             (setq box-idx (argmax box-sizes))
             (setq box-size (elt box-sizes box-idx))
             (setq box-sizes (remove box-size box-sizes))
             (when (string= (elt candidates obj-idx) target-object) (return))
             )
      (position box-size box-sizes-bak)))
  (:recognize-bin-boxes
    (&key (stamp (ros::time-now)))
    (let ((box-topic (format nil "transformable_bin_markers/output/boxes"))
          box-msg box-list bin-list)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[~a] [:recognize-bin-boxes] recognize bin boxes" (ros::get-name))
          (setq box-list (send box-msg :boxes))
          (setq bin-list (list :a :b :c :d :e :f :g :h :i :j :k :l))
          (dolist (bin bin-list)
            (sethash bin bin-boxes- (car box-list))
            (setq box-list (cdr box-list))))
        (ros::ros-fatal "[:recognize-bin-boxes] cannot recognize bin boxes"))))
  (:recognize-order-bin-box
    (arm &key (stamp (ros::time-now)))
    (let ((box-topic (format nil "~a_hand_camera/in_tote_clipper/output/box_array"
                             (arm2str arm)))
                     box-msg bbox)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[:recognize-order-bin-box] arm: ~a Recognize order bin box" (arm2str arm))
          (setq bbox (car (send box-msg :boxes)))
          (sethash arm order-bin-box- bbox))
        (ros::ros-fatal "[:recognize-order-bin-box] arm: ~a Cannot find order bin" (arm2str arm)))))
  (:bbox->cube
    (bbox)
    (let* ((dims (ros::tf-point->pos (send bbox :dimensions)))
           (bx (make-cube (elt dims 0) (elt dims 1) (elt dims 2))))
      (send bx :newcoords
            (send self :tf-pose->coords
                  (send bbox :header :frame_id)
                  (send bbox :pose)))
      bx))
  (:cube->cube-parallel-to-coords
    (cube &rest args)
    (let (copycoords target-ids cube-min-max-list)
      (send cube :worldcoords)
      ;; without this, body isn't updated
      (setq copycoords
            (send
              (if (member :coords args) (cadr (member :coords args)) *world-coords*)
              :copy-worldcoords))
      (cond ((member :include-original-cube args)
             (setq target-ids '(0 7))
             )
            ((member :included-in-original-cube args)
             (setq target-ids '(3 4))
             )
            )
      (dotimes (i 3)
        (dolist (id target-ids)
          (pushback
            (elt (sort
                   (mapcar
                     #'(lambda (x) (aref (send copycoords :inverse-transform-vector x) i))
                           (send cube :vertices)
                           )
                   #'<) id)
            cube-min-max-list)
          )
        )
      (if cube-min-max-list
        (let (bx)
          (setq bx
                (make-cube (- (elt cube-min-max-list 1) (elt cube-min-max-list 0))
                           (- (elt cube-min-max-list 3) (elt cube-min-max-list 2))
                           (- (elt cube-min-max-list 5) (elt cube-min-max-list 4)))
                )
          (send bx :newcoords copycoords)
          (send bx :locate
                (float-vector
                  (/ (+ (elt cube-min-max-list 1) (elt cube-min-max-list 0)) 2)
                  (/ (+ (elt cube-min-max-list 3) (elt cube-min-max-list 2)) 2)
                  (/ (+ (elt cube-min-max-list 5) (elt cube-min-max-list 4)) 2)
                  )
                :local)
          bx)
        nil)
      )
    )
  (:get-movable-region-for-tote
    (arm &key (offset (list 0 0 0)))
    (let (cube)
      (if (gethash arm order-bin-box-)
        (progn
          (setq cube (send self :bbox->cube (gethash arm order-bin-box-)))
          (sethash arm order-bin-movable-region-
                   (send self :cube->movable-region cube :offset offset)))
        (ros::ros-error "[:get-movable-region-for-tote] arm: ~a No order bin found. Do :recognize-order-bin-box first." (arm2str arm)))))
  (:cube->movable-region
    (cube &key (offset (list 0 0 0)))
    (let (movable-region vertex-values)
      (send cube :worldcoords)
      (dotimes (i 3)
        (setq vertex-values (mapcar #'(lambda (x) (aref x i)) (send cube :vertices)))
        (pushback (list
                    (+ (apply #'min vertex-values) (elt offset i))
                    (- (apply #'max vertex-values) (elt offset i)))
                  movable-region))
      movable-region))
  (:visualize-bins
    ()
    (let (bins)
      (dolist (b (send bin-boxes- :list-values))
        (let ((bin-inside (send self :bbox->cube b))
              (bin-outside (make-cube (+ (m->mm (send b :dimensions :x)))
                                      (+ (m->mm (send b :dimensions :y)) 30)
                                      (+ (m->mm (send b :dimensions :z)) 30)))
              (bin-model))
          (send bin-outside :newcoords (send bin-inside :copy-worldcoords))
          (send bin-outside :translate (float-vector 15 0 0) :world)
          (setq bin-model (body- bin-outside bin-inside))
          (send bin-model :set-color :blue 0.5)
          (pushback bin-model bins)
          ))
      bins))
  (:visualize-objects
    (&key (sib t))
    (let ((boxes-in-bin (if sib objects-sib-boxes- objects-in-bin-boxes-))
          (objs))
      (dolist (boxes (send boxes-in-bin :list-values))
        (dolist (b boxes)
          (let ((cube (send self :bbox->cube b)))
            (send cube :set-color :red)
            (pushback cube objs)
            )
          )
        )
      objs))
  (:recognize-grasp-coords-list
    (bin &key (stamp (ros::time-now)))
    (let (msg coords-list)
      (setq msg
        (one-shot-subscribe
          (format nil
                  "bin_~a_solidity_rag_merge_cpi_decomposer/centroid_pose_array"
                  (symbol2str bin))
          geometry_msgs::PoseArray
          :after-stamp stamp))
      (dolist (pose-msg (send msg :poses))
        (pushback (send self :tf-pose->coords (send msg :header :frame_id) pose-msg) coords-list)
        )
      (setq bin-coords-list- coords-list)
      )
    )
  (:recognize-objects-in-bin
    (bin &key (stamp (ros::time-now)) (timeout 10))
    (let ((box-topic (format nil "bin_~a_cpi_decomposer/boxes" (symbol2str bin)))
          box-msg
          (com-topic (format nil "bin_~a_cpi_decomposer/centroid_pose_array" (symbol2str bin)))
          com-msg)
      (ros::subscribe box-topic jsk_recognition_msgs::BoundingBoxArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq box-msg msg)))))
      (ros::subscribe com-topic geometry_msgs::PoseArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq com-msg msg)))))
      (while (not (and box-msg com-msg))
             (unix::usleep (* 50 1000))
             (ros::spin-once))
      (ros::unsubscribe box-topic)
      (ros::unsubscribe com-topic)
      (sethash bin objects-in-bin-boxes- (send box-msg :boxes))
      (sethash bin objects-in-bin-coms- com-msg)))
  (:recognize-objects-segmentation-in-bin
    (arm bin &key (stamp (ros::time-now)) (timeout 5))
    (let ((box-topic
            (format nil "~a_hand_camera/cluster_indices_decomposer_target/boxes" (arm2str arm)))
          box-msg
          (com-topic
            (format nil "~a_hand_camera/cluster_indices_decomposer_target/centroid_pose_array"
                        (arm2str arm)))
          com-msg obj-box obj-coords)
      (ros::subscribe box-topic jsk_recognition_msgs::BoundingBoxArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq box-msg msg)))))
      (ros::subscribe com-topic geometry_msgs::PoseArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq com-msg msg)))))
      (while (and
               (not (and box-msg com-msg))
               (> timeout (- (send (ros::time-now) :to-sec) (send stamp :to-sec))))
             (unix::usleep (* 50 1000))
             (ros::spin-once))
      (ros::unsubscribe box-topic)
      (ros::unsubscribe com-topic)
      (cond
        ((and box-msg com-msg)
         (ros::ros-info "[:recognize-objects-segmentation-in-bin] arm: ~a get sib com and bbox" arm)
         (setq obj-box (send box-msg :boxes))
         (setq obj-coords
               (mapcar #'(lambda (obj-pose)
                           (send self :tf-pose->coords
                                 (send com-msg :header :frame_id) obj-pose))
                       (send com-msg :poses)))
         (sethash bin objects-sib-boxes- obj-box)
         (sethash bin objects-sib-coords- obj-coords))
        (t
         (ros::ros-error "[:recognize-objects-segmentation-in-bin] arm: ~a failed to get sib msg" arm)
         (sethash bin objects-sib-boxes- nil)
         (sethash bin objects-sib-coords- nil)))))
  (:recognize-objects-in-order-bin
    (&key arm (stamp (ros::time-now)) (timeout 10))
    (let ((box-topic (format nil "~a_hand_camera/cluster_indices_decomposer_target/boxes" (arm2str arm)))
          box-msg
          (com-topic (format nil "~a_hand_camera/cluster_indices_decomposer_target/centroid_pose_array"
                        (arm2str arm)))
          com-msg obj-box obj-coords)
      (ros::subscribe box-topic jsk_recognition_msgs::BoundingBoxArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq box-msg msg)))))
      (ros::subscribe com-topic geometry_msgs::PoseArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq com-msg msg)))))
      (while (not (and box-msg com-msg))
             (unix::usleep (* 50 1000))
             (ros::spin-once))
      (ros::unsubscribe box-topic)
      (ros::unsubscribe com-topic)
      (cond
        ((and box-msg com-msg)
         (ros::ros-info "[:recognize-objects-in-order-bin] arm: ~a get cpi msg" arm)
         (setq obj-box (send box-msg :boxes))
         (setq obj-coords
               (mapcar #'(lambda (obj-pose)
                           (send self :tf-pose->coords
                                 (send com-msg :header :frame_id) obj-pose))
                       (send com-msg :poses)))
         (sethash arm objects-in-order-bin-boxes- obj-box)
         (sethash arm objects-in-order-bin-coords- obj-coords))
        (t
         (ros::ros-error "[:recognize-objects-in-order-bin] arm: ~a failed to get cpi msg" arm)))))
  (:recognize-objects-in-bin-loop
    (arm bin &key (max-try-num 2))
    (let (overlook-n-tried offset)
      (dotimes (overlook-n-tried max-try-num)
             (unix::sleep 10)
             (when (ros::get-param "~collect_data")
               (ros::set-param
                 (format nil "~a_collect_sib_data/ready_to_save" (send self :arm-symbol2str arm))
                 t)
               (unix::sleep 2)
               )
             (ros::ros-info "[recognize-objects-in-bin-loop] Segmentating objects in bin ~a" bin)
             (setq offset (gethash bin (gethash :default-overlook-offset hard-coded-variables-)))
             (send self :angle-vector (send self :bin-overlook-pose arm bin :offset offset) 3000)
             (if (send *ri* :recognize-objects-segmentation-in-bin
                           arm bin :stamp (ros::time-now))
               (return-from :recognize-objects-in-bin-loop t))
             (ros::ros-info
               "[recognize-objects-in-bin-loop] Segmentating in bin ~a failed  overlook-n-tried: ~a"
               bin overlook-n-tried)
             (dotimes (i 3) (setf (aref offset i) (+ (aref offset i) (random 10) -5)))
             (ros::ros-info
               "[recognize-objects-in-bin-loop] Change offset of overlook pose to ~a"
               offset)
             )
      nil)
    )
  (:recognize-object-in-hand
    (arm &key (stamp (ros::time-now)) (timeout))
    (let* (msg)
      (setq msg
            (one-shot-subscribe
              (format nil "~a_hand/object_verification/output" (arm-to-str arm))
              jsk_2015_05_baxter_apc::ObjectRecognition
              :after-stamp stamp
              :timeout timeout))
      (if msg
        (progn
          (ros::ros-info "[~a] [:recognize-object-in-hand] recognized as: ~a" (ros::get-name) (send msg :matched))
          (send msg :matched))
        nil)
      ))
  (:recognize-object-length-in-hand
    (arm &key (stamp (ros::time-now)) (timeout))
    (let (msg x-length)
      (setq msg
            (one-shot-subscribe
              (format nil "cpi_decomposer_~a_hand/boxes" (arm2str arm))
              jsk_recognition_msgs::BoundingBoxArray
              :after-stamp stamp
              :timeout timeout))
      (if (and msg (send msg :boxes))
        (progn
          (setq x-length
                (apply #'max
                       (mapcar #'(lambda (box) (send (send box :dimensions) :x))
                               (send msg :boxes))))
          (ros::ros-info "[:recognize-object-size-in-hand] length of object in hand : ~a" x-length)
          x-length)
        nil)))
  (:verify-object
    (arm object-name &key (stamp (ros::time-now)))
    (let ((req (instance std_srvs::TriggerRequest :init)))
      (when (ros::get-param "/apc2016_collect_data" nil)
        (ros::ros-info-green "[~a] [:verify-object] Saving image in hand" (ros::get-name))
        (while
          (not
            (send
              (ros::service-call (format nil "data_collection_server_~a/save_request" (arm-to-str arm)) req)
              :success))
          (unix::sleep 1)
          )
        )
      )
    (let (matched)
      (if (setq matched (send self :recognize-object-in-hand arm :stamp stamp :timeout 20000))
        (progn
          (ros::ros-info-green "[~a] [:verify-object] Must be ~a == ~a" (ros::get-name) object-name matched)
          (string= object-name matched))
        (progn
          (ros::ros-warn "[~a] [:verify-object] Timeout" (ros::get-name))
          :timeout))
      ))
  (:try-to-pick-in-bin
    (arm bin)
    (let* ((bin-box (gethash bin bin-boxes-))
           (dim-x (m->mm (send bin-box :dimensions :x)))
           (dim-y (m->mm (send bin-box :dimensions :y)))
           (dim-z (m->mm (send bin-box :dimensions :z)))
           (step-x (/ dim-x 4))
           (step-y (/ dim-y 4))
           (step-z (/ dim-z 4)))
      (send self :ik->bin-entrance arm bin
            :offset (float-vector
                      (* (random 4) step-x)
                      (- (* (random 4) step-y) (/ dim-y 2))
                      0))
      (send self :send-av 3000)
      ;; start vacuum gripper
      (send self :start-grasp arm)
      ;; grasp object
      (send *baxter* arm :move-end-pos #f(0 0 -80) :world :rotation-axis :z :revert-if-fail nil)
      (send self :send-av)
      (send self :wait-interpolation)
      ;; lift object
      (send *baxter* arm :move-end-pos #f(0 0 100) :world :rotation-axis :z :revert-if-fail nil)
      (send self :send-av)
      (send self :wait-interpolation)
      ))
  (:try-to-pick-object-solidity
    (arm bin &key (offset #f(0 0 0)))
    (let (pos-list centroid)
      (unless bin-coords-list-
        (ros::ros-error "no solidity coords-list is found")
        (return-from :try-to-pick-object-solidity))
      (dolist (c bin-coords-list-)
        (pushback (send c :worldpos) pos-list))
      (setq centroid
            (make-coords :pos (vec-list-max pos-list :axis 2)
                         :rpy (float-vector 0 0 0)))  ;; z axis max
      (setq offset-coords
            (send (send centroid :copy-worldcoords) :translate offset :world))
      (send self :angle-vector (send *baxter* arm :inverse-kinematics offset-coords) 3000)
      (send self :wait-interpolation)
      (send self :angle-vector (send *baxter* arm :inverse-kinematics centroid) 3000)
      (send self :wait-interpolation)
      ))
  (:try-to-pick-object
    (arm bin &key (object-index 0) (offset #f(0 0 0)) (use-sib t))
    (let (avs-before-approach obj-boxes obj-box obj-coms obj-com graspingp gripper-req-l
                              bin-box bin-y-l bin-z-l target-coords-with-offset target-coords)
      ;; validate
      (cond (use-sib
              (unless
                (setq obj-box (elt (gethash bin objects-sib-boxes-) object-index))
                (ros::ros-warn "[SIB] No bbox  is found: ~a, ~a" arm bin)
                (return-from :try-to-pick-object nil))
              ;; with Center of Mass
              (unless
                (setq obj-coords (elt (gethash bin objects-sib-coords-) object-index))
                (ros::ros-warn "[SIB] No com is found: ~a, ~a" arm bin)
                (return-from :try-to-pick-object nil))
              )
            (t
              (unless
                (setq obj-boxes (gethash bin objects-in-bin-boxes-))
                (ros::ros-warn "No object is found: ~a, ~a" arm bin)
                (return-from :try-to-pick-object nil))
              (unless
                (setq obj-box (elt obj-boxes object-index))
                (ros::ros-warn "The object is not found: ~a ~a ~a" arm bin object-index)
                (return-from :try-to-pick-object nil))
              (setq obj-coms-msg (gethash bin objects-in-bin-coms-))
              (setq obj-com (elt (send obj-coms-msg :poses) object-index))
              ;; ik to obj a bit distant
              ;; with Center of Mass
              (setq obj-coords (send self :tf-pose->coords
                                     (send obj-coms-msg :header :frame_id) obj-com))
              )
            )
      ;; set requisite length for gripper to enter bin
      (setq gripper-req-l (+ (gethash :pad-link-l hard-coded-variables-)
                             (/ (gethash :gripper-tube-t hard-coded-variables-) 2)))
      (unless
        (setq bin-box (gethash bin bin-boxes-))
        (ros::ros-error "[:try-to-pick-object] No data about bin-box ~a." bin)
        (ros::ros-error "[:try-to-pick-object] Call :recognize-bin-boxes first.")
        (return-from :try-to-pick-object nil))
      (let (bin-cube-base obj-cube-base)
        (setq bin-cube-base (send self :cube->cube-parallel-to-coords
                                  (send self :bbox->cube bin-box)
                                  :included-in-original-cube)
              obj-cube-base (send self :cube->cube-parallel-to-coords
                                  (send self :bbox->cube obj-box)
                                  :include-original-cube)
              bin-y-l (y-of-cube bin-cube-base)
              bin-z-l (z-of-cube bin-cube-base))
        ;; if object is higher than highest end-coords when gripper is 90 degrees
        (if (> (+ (elt (send obj-cube-base :pos) 2) (/ (z-of-cube obj-cube-base) 2))
               (- (+ (elt (send bin-cube-base :pos) 2) (/ (z-of-cube bin-cube-base) 2))
                  gripper-req-l
                  (gethash :offset-object-h hard-coded-variables-))
               )
          (cond ((< gripper-req-l
                    ;; if space gripper can enter exists on right
                    (- (- (elt (send obj-cube-base :pos) 1) (/ (y-of-cube obj-cube-base) 2))
                       (- (elt (send bin-cube-base :pos) 1) (/ (y-of-cube bin-cube-base) 2))
                       (gethash :offset-object-w hard-coded-variables-))
                    )
                 (setq sign 1)
                 (setq sign_y 0))
                ((< gripper-req-l
                    ;; if space gripper can enter exists on left
                    (- (+ (elt (send bin-cube-base :pos) 1) (/ (y-of-cube bin-cube-base) 2))
                       (+ (elt (send obj-cube-base :pos) 1) (/ (y-of-cube obj-cube-base) 2))
                       (gethash :offset-object-w hard-coded-variables-))
                    )
                 (setq sign -1)
                 (setq sign_y 0))
                (t
                  (setq sign 0)
                  (setq sign_y -1))
                )
          (progn
            (setq sign 0)
            (setq sign_y 0)
            )
          )
        (ros::ros-info "[:try-to-pick-object] arm:~a sign: ~a sign_y: ~a" arm sign sign_y)
        (setq target-coords-with-offset
              (make-coords
                :pos (v+ (send obj-coords :pos) offset)
                :rpy (list 0 (* sign_y pi/2) (* sign pi/2))
                ))
        (setq target-coords
              (make-coords
                :pos (send obj-coords :pos)
                :rpy (list 0 (* sign_y pi/2) (* sign pi/2))
                ))
        (let ((bin-reachable-limit-x
                (- (elt (send bin-cube-base :pos) 0) (/ (x-of-cube bin-cube-base) 2)
                   (- (send self :bin-reachable-depth (+ 90 (* sign_y 90)))))
                ))
          (when (> (elt (send target-coords-with-offset :worldpos) 0)
                   bin-reachable-limit-x)
            (ros::ros-warn
              "[:try-to-pick-object] arm:~a  cog of object is too deep to approach with sign_y: ~a"
              arm sign_y)
            (when (> (- (elt (send obj-cube-base :pos) 0) (/ (x-of-cube obj-cube-base) 2))
                     bin-reachable-limit-x)
              (ros::ros-warn
                "[:try-to-pick-object] arm:~a  whole object bbox is in unreachable zone"
                arm)
              (ros::ros-warn "[:try-to-pick-object] abort picking")
              (return-from :try-to-pick-object nil)
              )
            (setf (elt (send target-coords-with-offset :pos) 0) bin-reachable-limit-x)
            (setf (elt (send target-coords :pos) 0) bin-reachable-limit-x)
            (ros::ros-info
              "[:try-to-pick-object] arm:~a  change target position to ~a"
              arm (send target-coords :pos))
            )
          )
        )
      (if (ros::get-param "/apc_on_gazebo" nil)
        ;; FIXME: this condition is needed to avoid collision to the shelf at this moment
        (setq offset-from-entrance #f(-30 0 0))
        (if (< sign_y 0)
          (setq offset-from-entrance (float-vector -30 0 0))  ;; when gripper goes straight
          (setq offset-from-entrance
                (float-vector
                  (if (= sign 0) -30 (- (gethash :draw-out-from-bin-l hard-coded-variables-)))
                  0
                  (- (/ bin-z-l 2) gripper-req-l
                     (gethash :offset-gripper-bin-top hard-coded-variables-))
                  )
                )
          )
        )
      (pushback
        (send self :ik->bin-entrance arm bin
              :offset offset-from-entrance :gripper-angle (* (+ sign_y 1) 90))
        avs-before-approach)
      (if (and (not (= sign 0)) (>= sign_y 0))
        (let (end-coords bin-dim-x)
          (setq end-coords (send self :tf-pose->coords
                                 (send (send bin-box :header) :frame_id)
                                 (send bin-box :pose)))
          (setq bin-dim-x (m->mm (send (send bin-box :dimensions) :x)))
          (send end-coords :translate
                (float-vector
                  (- (+ (/ bin-dim-x 2) (gethash :draw-out-from-bin-l hard-coded-variables-)))
                  (* sign (- gripper-req-l (/ bin-y-l 2)
                             (- (gethash :offset-gripper-bin-side hard-coded-variables-))))
                  0)
                :world)
          (send end-coords :rotate (* sign pi/2) :x :local)
          ;; decide target coords depending on size of gripper and bin
          (pushback
            (send *baxter* arm :inverse-kinematics
                  end-coords
                  )
            avs-before-approach)
          (pushback
            (send *baxter* arm :move-end-pos
                  (float-vector
                    (gethash :draw-out-from-bin-l hard-coded-variables-)
                    0 0)
                  :world)
            avs-before-approach)
          (pushback
            (send *baxter* arm :move-end-pos
                  (float-vector
                    0 0
                    (+ (gethash :offset-gripper-bin-side hard-coded-variables-) 10)
                    )
                  :local)
            ;; press back of gripper against bin wall
            avs-before-approach)
          )
        )
      ;; check if IK to object succeed
      ;; if that IK fails, abort picking
      (if (send *baxter* arm :inverse-kinematics target-coords)
        (progn
          (send self :angle-vector-sequence avs-before-approach :fast nil 0 :scale 5.0)
          (send self :wait-interpolation)
          ;; revert *baxter* before IK to object
          (send *baxter* :angle-vector (car (reverse avs-before-approach)))
          (when (and (not (= sign 0)) (>= sign_y 0))
            ;; stop pressing and detach gripper from wall
            (send self :state)
            (send *baxter* :angle-vector (send self :potentio-vector))
            (send self :angle-vector (send *baxter* arm :move-end-pos #f(0 0 -20) :local) 2000)
            (send self :wait-interpolation)
            )
          (setq last-av-before-approach- (send *baxter* :angle-vector))
          ;; grasp object
          (ros::ros-info "[:try-to-pick-object] arm:~a approach to the object" arm)
          (if (< sign_y 0)
            ;; when gripper is straight
            ;; move arm to the target only if robot can solve IK
            (if (send *baxter* arm :inverse-kinematics target-coords-with-offset)
              (progn
                (send self :angle-vector (send *baxter* :angle-vector) 3000)
                (send self :wait-interpolation-smooth 1000)
                )
              (ros::ros-warn
                "[:try-to-pick-object] arm:~a cannot approach closer with offset" arm)
              )
            ;; when gripper isn't straight
            (let (end-coords end->above target-above-coords (elapsed-t 0))
              (setq end-coords (send (send *baxter* arm :end-coords) :copy-worldcoords))
              (setq end->above (send end-coords :transformation target-coords-with-offset :local))
              (setf (elt (send end->above :pos) 2) 0)
              (setq target-above-coords
                    (send end-coords :transform end->above :local))
              ;; change gripper-angle to 45 not to push target object
              (send *baxter* :rotate-gripper arm 45 :relative nil)
              (send self :angle-vector (send *baxter* :angle-vector) 1000)
              (send *ri* :wait-interpolation)
              (send *baxter* :rotate-gripper arm 90 :relative nil)
              ;; move arm in the local x, y direction only if robot can solve IK
              (if (send *baxter* arm :inverse-kinematics target-above-coords
                        :revert-if-fail t)
                (progn
                  (send *baxter* :rotate-gripper arm 45 :relative nil)
                  (send self :angle-vector (send *baxter* :angle-vector)
                        (setq elapsed-t 2500))
                  (send self :wait-interpolation)
                  (send *baxter* :rotate-gripper arm 90 :relative nil)
                  (send self :angle-vector (send *baxter* :angle-vector) 1000)
                  )
                )
              (send self :wait-interpolation)
              ;; move arm to the target
              ;; if above IK succeeded, arm moves in the local z direction
              (if (send *baxter* arm :inverse-kinematics target-coords-with-offset)
                (progn
                  (send self :angle-vector
                        (send *baxter* :angle-vector)
                        (- 4000 elapsed-t))
                  (send self :wait-interpolation-smooth 1000)
                  )
                (ros::ros-warn
                  "[:try-to-pick-object] arm:~a cannot approach closer with offset" arm)
                )
              )
            )
          ;; move arm to the target only if robot can solve IK
          (if (send *baxter* arm :inverse-kinematics target-coords)
            (progn
              ;; start the vacuum gripper after approaching to the object
              (ros::ros-info "[:try-to-pick-object] arm:~a start vacuum gripper" arm)
              (send self :start-grasp arm)
              (unix::sleep 1)
              (send self :angle-vector (send *baxter* :angle-vector) 3000)
              (send self :wait-interpolation)
              (setq graspingp (send self :graspingp arm))
              (ros::ros-info "[:try-to-pick-object] arm:~a graspingp: ~a" arm graspingp)
              (unless graspingp
                (ros::ros-info "[:try-to-pick-object] arm:~a again approach to the object" arm)
                (let ((temp-av (send *baxter* :angle-vector)))
                  ;; only if robot can solve IK
                  (if (send *baxter* arm :move-end-pos #f(0 0 -50) :local)
                    (send self :angle-vector (send *baxter* :angle-vector) 3000)
                    (ros::ros-warn "[:try-to-pick-object] arm:~a fail to again approach" arm))
                  (send self :wait-interpolation)
                  ;; revert baxter
                  (send self :angle-vector (send *baxter* :angle-vector temp-av) 3000)
                  (send self :wait-interpolation)
                  )
                )
              ;; lift object
              (ros::ros-info "[:try-to-pick-object] arm:~a lift the object" arm)
              (cond ((< sign_y 0)
                     (send self :gripper-servo-off arm)
                     ;; only if robot can solve IK
                     (if (send *baxter* arm :move-end-pos #f(0 0 40) :world)
                       (send self :angle-vector (send *baxter* :angle-vector) 3000)
                       (ros::ros-warn "[:try-to-pick-object] arm:~a fail to lift the object" arm)))
                    (t
                      ;; only if robot can solve IK
                      (if (send *baxter* arm :move-end-pos #f(0 0 80) :local)
                        (progn
                          (if (not (= sign 0))
                            (send *baxter* arm :move-end-pos #f(0 0 30) :world :revert-if-fail t))
                          (send self :angle-vector (send *baxter* :angle-vector) 3000)
                          )
                        (ros::ros-warn "[:try-to-pick-object] arm:~a fail to lift the object" arm)))
                    )
              (send self :wait-interpolation)
              (unix::sleep 1)  ;; wait for arm to follow
              )
            (progn
              (ros::ros-warn
                "[:try-to-pick-object] arm:~a cannot approach closer without offset" arm)
              (ros::ros-warn "[:try-to-pick-object] arm:~a abort picking" arm)
              )
            )
          )
        (progn
          (ros::ros-warn "[:try-to-pick-object] arm:~a cannot approach closer without offset" arm)
          (ros::ros-warn "[:try-to-pick-object] arm:~a abort picking" arm)
          (send *baxter* :angle-vector last-av-before-approach-)  ;; revert *baxter*
          )
        )
      (setq graspingp (send self :graspingp arm))
      (ros::ros-info "[:try-to-pick-object] arm:~a graspingp: ~a" arm graspingp)
      graspingp))
  (:pick-object
    (arm bin &key (object-index 0) (n-trial 1) (n-trial-same-pos 1) (do-stop-grasp nil) (use-sib t)
         (target-offset #f(0 0 0)))
    (send *ri* :angle-vector-sequence
          (list (send self :ik->bin-entrance arm bin
                      :offset (float-vector
                                -150 0
                                (gethash :offset-avoid-bin-top hard-coded-variables-)
                                )
                      ))
          :fast nil 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    ;; initialize last-av-before-approach-
    (setq last-av-before-approach- (send *baxter* :angle-vector))
    (let (bin-box graspingp avs)
      ;; abort when no objects found
      (if use-sib
        (if
          (or
            (eq (length (gethash bin objects-sib-boxes-)) 0)
            (eq (length (gethash bin objects-sib-coords-)) 0))
          (return-from :pick-object nil) t)
        (if
          (eq (length (gethash bin objects-in-bin-boxes-)) 0)
          (return-from :pick-object nil) t)
        )
      ;; move arm bin-entrance -> pos above object to z direction
      (setq bin-box (gethash bin bin-boxes-))
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (unless graspingp
            (setq graspingp
                  (send self :try-to-pick-object arm bin :object-index object-index
                        :offset (v+ (float-vector (* i 50) 0 0) target-offset) :use-sib use-sib)
                  )
            (pushback (send *baxter* :angle-vector) avs)
            )
          (when (and graspingp (> (send self :real-sim-end-coords-diff arm) 80))
            (ros::ros-info
              "arm:~a Grasped wall so stop-grasp, diff: ~a" arm
              (send self :real-sim-end-coords-diff arm))
            (send self :stop-grasp arm)
            (unix::sleep 3)
            )
          )
        (when (and graspingp (> (send self :real-sim-end-coords-diff arm) 80))
          (ros::ros-info
            "arm:~a Grasped wall so stop-grasp, diff: ~a" arm
            (send self :real-sim-end-coords-diff arm))
          (send self :stop-grasp arm)
          (unix::sleep 3)
          )
        )
      (when do-stop-grasp (unless graspingp (send self :stop-grasp arm)))
      (send self :angle-vector-sequence (reverse avs) :fast nil 0 :scale 5.0)
      (send self :wait-interpolation)
      (ros::ros-info "[:pick-object] arm:~a in-bin -> bin-entrance" arm)
      ;; move arm in-bin -> bin-entrance
      (unless graspingp
        (send *baxter* :rotate-gripper arm 0 :relative nil)
        (send self :angle-vector (send *baxter* :angle-vector) 1000))
      (send *baxter* :angle-vector last-av-before-approach-)
      (unless graspingp
        (send *baxter* :rotate-gripper arm 0 :relative nil))
      (send self :angle-vector-sequence
            (list
              (send *baxter* :angle-vector)
              (send *baxter* arm :move-end-pos
                           (float-vector
                             (- (gethash :draw-out-from-bin-l hard-coded-variables-))
                             0 0)
                           :world
                           :rotation-axis :z
                           :revert-if-fail nil
                           )
              (send *baxter* :rotate-gripper arm 90 :relative nil)
              (send self :ik->bin-entrance arm bin
                           :offset (float-vector
                                     -100 0 (gethash :offset-avoid-bin-top hard-coded-variables-))
                           )
              (send self :ik->bin-entrance arm bin
                           :offset (float-vector
                                     -200 0 (gethash :offset-avoid-bin-top hard-coded-variables-))
                           )
              )
            :fast nil 0 :scale 5.0)
      (send self :wait-interpolation)
      ;; move arm bin-entrance -> body
      (send *baxter* :avoid-shelf-pose arm bin)
      (send self :send-av)
      (send self :wait-interpolation)
      (send self :gripper-servo-on arm)
      (unix::sleep 1)
      (setq graspingp (send self :graspingp arm))
      graspingp))
  (:pick-object-in-order-bin
    (arm &key (n-trial 1) (n-trial-same-pos 1) (do-stop-grasp nil) (object-index nil))
    (let (order-bin-box graspingp avs avs->view)
      (if (or
          (null (gethash arm objects-in-order-bin-boxes-))
          (null (gethash arm objects-in-order-bin-coords-)))
        (return-from :pick-object-in-order-bin nil) t)
      ;; (send self :fold-pose-back arm)
      ;; (send self :wait-interpolation)
      (unless object-index
        (setq object-index (random (length (gethash arm objects-in-order-bin-boxes-)))))
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (unless graspingp
            (setq graspingp
                  (send self :try-to-pick-object-in-order-bin arm
                        :offset (float-vector 0 0 (- (* i -50) 30))
                        :object-index object-index))
            (pushback (send *baxter* :angle-vector) avs))))
      (when do-stop-grasp (unless graspingp (send self :stop-grasp arm)))
      (send self :angle-vector-sequence-raw (reverse avs) :fast nil 0 :scale 5.0)
      (send self :wait-interpolation)
      (ros::ros-info "[:pick-object] arm:~a in-order-bin -> view-hand-pose" arm)
      (send self :gripper-servo-on arm)
      (pushback (send *baxter* arm :move-end-pos #f(0 0 150) :world) avs->view)
      (if (eq arm :larm)
        (pushback (send *baxter* :view-hand-pose arm :j) avs->view)
        (pushback (send *baxter* :view-hand-pose arm :l) avs->view))
      (send *ri* :angle-vector-sequence-raw avs->view :fast nil 0 :scale 5.0)
      (send self :wait-interpolation)
      graspingp))
  (:try-to-pick-object-in-order-bin
    (arm &key (offset #f(0 0 0)) (object-index 0))
    (let (avs obj-box graspingp order-bin-box movable-region obj-pos obj-box-z-length)
      ;; validate
      (unless
        (setq obj-box (elt (gethash arm objects-in-order-bin-boxes-) object-index))
        (ros::ros-warn "[:try-to-pick-object-in-order-bin] No bbox  is found: ~a, ~a" arm bin)
        (return-from :try-to-pick-object-in-order-bin nil))
      ;; with Center of Mass
      (unless
        (setq obj-coords (elt (gethash arm objects-in-order-bin-coords-) object-index))
        (ros::ros-warn "[:try-to-pick-in-order-bin] No com is found: ~a, ~a" arm bin)
        (return-from :try-to-pick-object-in-order-bin nil))
      (unless
        (setq order-bin-box (gethash arm order-bin-box-))
        (ros::ros-error "[:try-to-pick-object] No data about order bin box. Call :recognize-order-bin-box first.")
        (return-from :try-to-pick-object-in-order-bin nil))
      (unless
        (setq movable-region (gethash arm order-bin-movable-region-))
        (ros::ros-error "[:try-to-pick-object] No data about movable region for order bin box. Call :get-movable-region-for-tote first.")
        (return-from :try-to-pick-object-in-order-bin nil))
      (setq sign 0)
      (setq sign_y -1)
      (ros::ros-info "[:try-to-pick-object-in-order-bin] arm:~a sign: ~a sign_y: ~a" arm sign sign_y)
      ;; grasp object
      (ros::ros-info "[:try-to-pick-object-in-order-bin] arm:~a approach to the object" arm)
      ;; gripper is straight
      (send self :gripper-servo-on arm)
      (setq obj-pos (send obj-coords :worldpos))
      (dolist (i (list 0 1))
        (cond ((> (elt (elt movable-region i) 0) (aref obj-pos i))
               (ros::ros-info "[:try-to-pick-in-order-bin] object is out of movable region. ~a > ~a < ~a"
                              (elt (elt movable-region i) 1)
                              (elt (send obj-coords :pos) i)
                              (elt (elt movable-region i) 0))
               (setf (aref obj-pos i) (elt (elt movable-region i) 0)))
              ((> (aref obj-pos i) (elt (elt movable-region i) 1))
               (ros::ros-info "[:try-to-pick-in-order-bin] object is out of movable region. ~a < ~a > ~a"
                              (elt (elt movable-region i) 1)
                              (elt (send obj-coords :pos) i)
                              (elt (elt movable-region i) 0))
               (setf (aref obj-pos i) (elt (elt movable-region i) 1)))
              (t nil)))
      (setq obj-box-z-length (z-of-cube (send self :bbox->cube obj-box)))
      (setq obj-pos (v+ obj-pos (float-vector 0 0 (/ obj-box-z-length 2))))
      (send self :angle-vector
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos (v+ obj-pos offset)
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            3000)
      (send self :wait-interpolation)
      ;; start the vacuum gripper after approaching to the object
      (ros::ros-info "[:try-to-pick-object-in-order-bin] arm:~a start vacuum gripper" arm)
      (send self :start-grasp arm)
      (unix::sleep 1)
      (send self :angle-vector
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos obj-pos
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            3000)
      (send self :wait-interpolation)
      (setq graspingp (send self :graspingp arm))
      (ros::ros-info "[:try-to-pick-object-in-order-bin] arm:~a graspingp: ~a" arm graspingp)
      (unless graspingp
        (ros::ros-info "[:try-to-pick-object-in-order-bin] arm:~a again approach to the object" arm)
        (let ((temp-av (send *baxter* :angle-vector)))
          ;; only if robot can solve IK
          (if (send *baxter* arm :move-end-pos #f(0 0 -50) :local)
            (send self :angle-vector (send *baxter* :angle-vector) 3000))
          (send self :wait-interpolation)
          (send self :angle-vector (send *baxter* :angle-vector temp-av) 3000)  ;; revert baxter
          (send self :wait-interpolation)))
      ;; lift object
      (ros::ros-info "[:try-to-pick-object-in-order-bin] arm:~a lift the object" arm)
      (send self :gripper-servo-off arm)
      (send self :angle-vector (send *baxter* arm :move-end-pos #f(0 0 200) :world) 3000)
      (send self :wait-interpolation)
      (unix::sleep 1)  ;; wait for arm to follow
      (setq graspingp (send self :graspingp arm))
      (ros::ros-info "[:try-to-pick-object-in-order-bin] arm:~a graspingp: ~a" arm graspingp)
    graspingp))
  (:send-av
    (&optional (tm 3000) (ctype nil))
    (send self :angle-vector (send *baxter* :angle-vector) tm ctype))
  (:force-to-reach-goal (&key (arm :arms) (threshold 5) (stop 10))
    (let ((times 0) arms)
      (case arm
        (:arms (setq arms '(:larm :rarm)))
        (:larm (setq arms '(:larm)))
        (:rarm (setq arms '(:rarm)))
        )
      (dolist (arm arms)
        (while
          (> (norm (v- (send *baxter* arm :angle-vector)
                       (send *ri* :arm-potentio-vector arm))) threshold)
          (when (> times stop) (ros::ros-warn "abort to send avs: ~A" arm) (return))  ;; abort
          (ros::ros-info "difference of avs: ~A"
                         (norm (v- (send *baxter* arm :angle-vector)
                                   (send *ri* :arm-potentio-vector arm))))
          (send *ri* :angle-vector (send *baxter* :angle-vector))
          (send *ri* :wait-interpolation)
          (incf times)
          )
        ))
    )
  (:ik->bin-entrance
    (arm bin &key (offset #f(0 0 0)) (gripper-angle 90)
         (rotation-axis :z) (overlook-angle 0) (revert-if-fail nil) (use-current-pose nil))
    (let ((bin-box)
          (bin-coords)
          (bin-dim-x)
          (nearest-pose-candidates
            (list
              (send *baxter* :fold-pose-upper arm)
              (send *baxter* :fold-pose-middle arm)
              (send *baxter* :fold-pose-lower arm)
              (send *baxter* :avoid-shelf-pose arm bin))))
      (setq bin-box (gethash bin bin-boxes-))
      (unless bin-box
        (ros::ros-error "[:ik->bin-entrance] No data about bin-box ~a." bin)
        (ros::ros-error "[:ik->bin-entrance] Call :recognize-bin-boxes first.")
        (return-from :ik->bin-entrance))
      (setq bin-coords (send self :tf-pose->coords
                             (send (send bin-box :header) :frame_id)
                             (send bin-box :pose)))
      (setq bin-dim-x (m->mm (send (send bin-box :dimensions) :x)))
      (send bin-coords :translate (float-vector (- (/ bin-dim-x 2)) 0 0) :world)
      (if (not use-current-pose)
        (send *baxter* :avoid-shelf-pose arm bin))
      (send *baxter* :ik->nearest-pose arm bin-coords nearest-pose-candidates :rotation-axis t)
      ;; apply rotation
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (send bin-coords :rotate (deg2rad (- gripper-angle 90)) :y :world)
      ;; apply offset
      (send bin-coords :translate offset :world)
      (send bin-coords :rotate overlook-angle :y :world)
      (ros::ros-info "gripper-angle: ~a" gripper-angle)
      (cond ((< gripper-angle 89)
             (send *baxter* :ik->nearest-pose arm bin-coords
                   nil
                   :rotation-axis t
                   ;; if this is :z when gripper is straight, wrist sometimes rotates overly
                   ;; if this is :z when Bin is h, body and arm collides on pulling out arm
                   :revert-if-fail revert-if-fail)
             )
            ((eq bin :h)
             (send *baxter* :ik->nearest-pose arm bin-coords
                   nearest-pose-candidates
                   :rotation-axis t
                   ;; if this is :z when gripper is straight, wrist sometimes rotates overly
                   ;; if this is :z when Bin is h, body and arm collides on pulling out arm
                   :revert-if-fail revert-if-fail)
             )
            (t
              (send *baxter* :ik->nearest-pose arm bin-coords
                    nearest-pose-candidates
                    :rotation-axis rotation-axis
                    :revert-if-fail revert-if-fail)
              )
            )
      )
    )
  (:rotate-wrist-ik
    (arm target-coords &rest args)
    (let (pose-candidates min-av max-av)
      (setq pose-candidates (list (send *baxter* :angle-vector)))
      (send *baxter* arm :wrist-r :joint-angle (send *baxter* arm :wrist-r :min-angle))
      (setq min-av (send *baxter* :angle-vector))
      (send *baxter* arm :wrist-r :joint-angle (send *baxter* arm :wrist-r :max-angle))
      (setq max-av (send *baxter* :angle-vector))
      (dolist (p '(0 0.25 0.5 0.75 1))
        (pushback (midpoint p min-av max-av) pose-candidates))
      (send *baxter* :angle-vector (car pose-candidates))
      (send* *baxter* :ik->nearest-pose arm target-coords pose-candidates args)
      )
    )
  (:move-arm-body->bin
    (arm bin &key (scale 3.0))
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :avoid-shelf-pose arm bin) avs)
      (pushback (send self :ik->bin-entrance arm bin
                      :offset (float-vector
                                -150 0 (gethash :offset-avoid-bin-top hard-coded-variables-))
                      )
                avs)
      (send self :angle-vector-sequence avs :fast nil 0 :scale scale)
      ))
  (:move-arm-body->bin-with-support-arm
    (arm bin distance &key (scale 3.0))
    (let (avs support-arm)
         (send *baxter* :avoid-shelf-pose arm bin)
         (send *baxter* (send self :opposite-arm arm) :inverse-kinematics
               (send *baxter* :get-support-arm-coords arm :dz (- distance))
               :use-gripper t
               :rotation-axis :z
               :revert-if-fail nil)
         (pushback (send *baxter* :angle-vector) avs)
         (send self :ik->bin-entrance arm bin
               :offset (float-vector
                         -150 0 (gethash :offset-avoid-bin-top hard-coded-variables-)))
         (send *baxter* (send self :opposite-arm arm) :inverse-kinematics
               (send *baxter* :get-support-arm-coords arm :dz (- distance))
               :use-gripper t
               :rotation-axis :z
               :revert-if-fail nil)
         (pushback (send *baxter* :angle-vector) avs)
         (send self :angle-vector-sequence avs :fast nil 0 :scale scale)))
  (:bin-overlook-pose
    (arm bin &key offset angle)
    (if (not offset)
      (setq offset (gethash bin (gethash :default-overlook-offset hard-coded-variables-))))
    (if (not angle)
      (setq angle (gethash :default-overlook-angle hard-coded-variables-)))
    (let (av)
      (setq av
            (send self :ik->bin-entrance arm bin
                  :offset offset
                  :rotation-axis t
                  :gripper-angle 90
                  :overlook-angle angle
                  :revert-if-fail t)
            )
      (if av
        av
        (send self :ik->bin-entrance arm bin
              :offset offset
              :rotation-axis :z
              :gripper-angle 90
              :overlook-angle angle
              :revert-if-fail nil)
        )
      )
    )
  (:move-arm-body->bin-overlook-pose
    (arm bin &key (scale 4.0))
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :avoid-shelf-pose arm bin) avs)
      (pushback (send self :ik->bin-entrance arm bin
                      :offset (float-vector
                                -200 0 (gethash :offset-avoid-bin-top hard-coded-variables-))
                      ) avs)
      (pushback (send self :bin-overlook-pose arm bin) avs)
      (send self :angle-vector-sequence avs :fast nil 0 :scale scale)
      (send self :wait-interpolation)
      ))
  (:move-arm-body->order-bin
    (arm &optional (ctype nil))
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :place-object-pose arm) avs)
      (send self :angle-vector-sequence avs :fast ctype 0 :scale 4.0)
      ))
  (:move-arm-body->order-bin-overlook-pose
    (arm)
    (let (avs)
      (send *ri* :angle-vector (send *baxter* :rotate-gripper arm 90 :relative nil))
      (if (eq arm :rarm)
        (pushback (send *baxter* arm :inverse-kinematics
                      (make-coords :pos #f(600 -150 -50)
                                   :rpy (list 0 pi/2 0)))
                avs)
        (pushback (send *baxter* arm :inverse-kinematics
                      (make-coords :pos #f(600 150 -50)
                                   :rpy (list 0 pi/2 0)))
                avs))
      (send self :angle-vector-sequence avs :fast nil 0 :scale 3.0)))
  (:spin-off-by-wrist
    (arm &key (times 10))
    (let (avs)
      (pushback (send *baxter* :angle-vector) avs)
      (dotimes (i times)
        (pushback (send *baxter* :rotate-wrist arm 10) avs)
        (pushback (send *baxter* :rotate-wrist arm -20) avs)
        (pushback (send *baxter* :rotate-wrist arm 20) avs)
        (pushback (send *baxter* :rotate-wrist arm -10) avs)
        )
      (send self :angle-vector-sequence-raw avs)
      ))
  (:move-arm-body->head-view-point
    (arm)
    (let (avs coords-tmp)
      (setq coords-tmp
            (if (eq arm :larm)
              (make-coords :pos #f(690.508 171.959 1034.31) :rpy #f(-1.0497 -0.422017 -1.71354))
              (make-coords :pos #f(649.331 -319.89 1028.69) :rpy #f(1.0497 -0.422017 1.71354))))
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* arm :inverse-kinematics coords-tmp :rotation-axis nil) avs)
      (pushback (send *baxter* arm :inverse-kinematics coords-tmp :rotation-axis t) avs)
      (send self :angle-vector-sequence avs :fast nil 0 :scale 4.0)
      ))
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
  (:hold-opposite-hand-object
    (arm distance)
    (let (avs)
      (setq avs
            (list
              (send *baxter* :approaching-from-downside-pose arm)
              (send *baxter* arm :inverse-kinematics
                    (send *baxter* :get-support-arm-coords (send self :opposite-arm arm)
                          :dz (- (+ distance 100)))
                    :use-gripper t
                    :rotation-axis :z)
              (send *baxter* arm :inverse-kinematics
                    (send *baxter* :get-support-arm-coords (send self :opposite-arm arm)
                          :dz (- distance))
                    :use-gripper t
                    :rotation-axis :z
                    :revert-if-fail nil)))
      (send self :angle-vector-sequence avs :fast nil 0 :scale 5.0)))
  (:place-object
    (arm)
    (send self :move-arm-body->order-bin arm)
    (send self :gripper-servo-off arm)
    (send self :wait-interpolation)
    (send self :stop-grasp arm)
    (send self :spin-off-by-wrist arm :times 20)
    (send self :gripper-servo-on arm)
    (send self :wait-interpolation)
    (send self :angle-vector (send *baxter* :fold-pose-back arm) 3000)
    (send self :wait-interpolation)
    )
  (:get-work-orders
    (arm)
    (setq
      msg
      (one-shot-subscribe
        (format nil "/strategic_work_order/~a_hand" (send self :arm-symbol2str arm))
        jsk_2015_05_baxter_apc::WorkOrderArray))
    (send msg :array))
  (:get-next-work-order
    (arm current-order)
    (let ((orders (send self :get-work-orders arm)))
      (when (eq (length orders) 0)
        (ros::ros-error "[:get-next-work-order] There is no order")
        (return-from :get-next-work-order nil))
      (when (null current-order) (return-from :get-next-work-order (elt orders 0)))
      (dotimes (i (- (length orders) 1))
        (when (string= (send (elt orders i) :bin) (send current-order :bin))
          (return-from :get-next-work-order (elt orders (+ i 1)))))))
  (:get-certain-work-order
    (arm bin)
    (let ((orders (send self :get-work-orders arm)))
      (when (eq (length orders) 0)
        (ros::ros-error "[:get-certain-work-order] There is no order")
        (return-from :get-certain-work-order nil))
      (when (null bin) (return-from :get-certain-work-order (elt orders 0)))
      (dotimes (i (length orders))
        (when (string= (send (elt orders i) :bin) (symbol2str bin))
          (return-from :get-certain-work-order (elt orders i))))))
  (:check-bin-exist
    (bin)
    (if (gethash bin bin-boxes-) t nil))
  (:get-bin-contents
    (bin)
    (ros::get-param
      (concatenate string
                   "/bin_contents/bin_"
                   (symbol-string bin))))
  (:get-tote-contents ()
    (ros::get-param "/tote_contents"))
  (:real-sim-end-coords-diff
    (arm)
    (let (real-coords sim-coords diff-coords thresh)
      (send self :update-robot-state)
      (setq real-coords (send (send robot arm :root-link)
                              :transformation (send robot arm :end-coords)))
      (setq sim-coords (send (send *baxter* arm :root-link)
                             :transformation (send *baxter* arm :end-coords)))
      (setq diff-coords (send real-coords :transformation sim-coords :local))
      (elt (send diff-coords :pos) 2)))
  (:wait-for-user-input-to-start
    (arm)
    """wait for user input to start"""
    (let (can-start)
      (ros::ros-info "[:wait-for-user-input-to-start] wait for user input to start: ~a" arm)
      (ros::wait-for-service "/rviz/yes_no_button")
      (while
        (not can-start)
        (setq can-start (send (ros::service-call
                                "/rviz/yes_no_button" (instance jsk_gui_msgs::YesNoRequest)) :yes))
        )
      (ros::ros-info "[:wait-for-user-input-to-start] received user input: ~a" arm)
      ))
  (:calib-pressure-threshold
    (&optional (arm :arms))
    (send self :start-grasp arm)
    (dolist (l/r (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (let ((min-pressure)
            (topic (format nil "/gripper_front/limb/~a/pressure/state" (arm-to-str l/r)))
            (hashkey (if (eq l/r :larm) :l-pressure-threshold :r-pressure-threshold)))
        (dotimes (i 7)
          (let ((pressure (send (one-shot-subscribe topic std_msgs::Float64) :data)))
            (when (or (null min-pressure) (< pressure min-pressure)) (setq min-pressure pressure))
            )
          (unix::sleep 1)
          )
        (sethash hashkey hard-coded-variables- (- min-pressure (if (eq l/r :larm) 5 10)))
        )
      )
    (send self :stop-grasp arm)
    (ros::ros-info "[:calib-pressure-threshold] Threshold r:~a l:~a"
                   (gethash :r-pressure-threshold hard-coded-variables-)
                   (gethash :l-pressure-threshold hard-coded-variables-)))
 ;; defmethod :select-stow-target-bin
 ;; randomly choose bin for stow task
  (:select-stow-target-bin
    (arm &key (object-length) (object-length-limit 0.1))
    (let (bin target-bin)
      (setq target-bin
            (car
              (delete nil (mapcar
                            #'(lambda (work-order)
                                (setq bin (str2symbol (send work-order :bin)))
                                (if (eq bin :tote)
                                  bin
                                  (if (and object-length
                                         (> object-length object-length-limit))
                                    (if (member bin (list :a :b :c :k :l :j)) bin nil)
                                    (if (not (eq bin :k)) bin nil))))
                        (send self :get-work-orders arm)))))
      (if (not target-bin)
        (if (eq arm :larm)
          (setq target-bin (elt (list :a :b :d :e :g :k) (random 5)))
          (setq target-bin (elt (list :c :f :h :i :k :l) (random 5)))))
      target-bin))
  )

(defclass jsk_2016_01_baxter_apc::baxter-moveit-environment
  :super moveit-environment)
(defmethod jsk_2016_01_baxter_apc::baxter-moveit-environment
  (:init (&key ((:robot rb) (instance jsk_2016_01_baxter_apc::baxter-robot :init)) &rest args)
         (send-super* :init :robot rb :frame-id "world" args))
  (:default-configuration ()
   (list (list :rarm
               (cons :group-name "right_arm")
               (cons :target-link
                     (send self :search-link-from-name "right_gripper_vacuum_pad"))
               (cons :joint-list (send robot :rarm :joint-list))
               )
         (list :larm
               (cons :group-name "left_arm")
               (cons :target-link
                     (send self :search-link-from-name "left_gripper_vacuum_pad"))
               (cons :joint-list (send robot :larm :joint-list)))
         (list :arms
               ;; can not use inverse-kinematics
               ;; currently only supports angle-vector
               (cons :group-name "both_arms")
               (cons :target-link
                     (list
                       (send self :search-link-from-name "left_gripper_vacuum_pad")
                       (send self :search-link-from-name "right_gripper_vacuum_pad")))
               (cons :joint-list (append (send robot :larm :joint-list)
                                         (send robot :rarm :joint-list)))))))

(defun jsk_2016_01_baxter_apc::baxter-init (&key (ctype :default-controller) (moveit nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance jsk_2016_01_baxter_apc::baxter-moveit-environment))
      (setq mvit-rb (instance jsk_2016_01_baxter_apc::baxter-robot :init)))
    (unless (boundp '*ri*)
      (setq *ri* (instance jsk_2016_01_baxter_apc::baxter-interface :init :type ctype
                           :moveit-environment mvit-env
                           :moveit-robot mvit-rb)))
    (unless (boundp '*baxter*)
      (setq *baxter* (instance jsk_2016_01_baxter_apc::baxter-robot :init)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)
  ))
