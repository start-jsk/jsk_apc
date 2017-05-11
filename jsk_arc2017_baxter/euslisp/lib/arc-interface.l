;; -*- mode: lisp;-*-

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/util.l")

(ros::load-ros-manifest "jsk_arc2017_baxter")

(unless (find-package "JSK_ARC2017_BAXTER")
  (make-package "JSK_ARC2017_BAXTER"))

;; TODO: Do not inherit baxter-interface in arc-interface
;; TODO: baxter-interface ---> arc-interface -> pick-interface
;; TODO: baxter-robot     -^
;; TODO: https://github.com/start-jsk/jsk_apc/issues/2068
(defclass jsk_arc2017_baxter::arc-interface
  :super jsk_2016_01_baxter_apc::baxter-interface
  :slots (object-boxes-in-bin-
          object-coords-in-bin-
          bin-movable-regions-))

(defmethod jsk_arc2017_baxter::arc-interface
  (:init
    (&rest args &key
           ((:moveit-environment mvit-env)
            (instance jsk_2016_01_baxter_apc::baxter-moveit-environment))
           ((:moveit-robot mvit-rb) (instance jsk_2016_01_baxter_apc::baxter-robot :init))
           &allow-other-keys)
    (send-super* :init :moveit-environment mvit-env
                 :moveit-robot mvit-rb args)
    ;; initialize slots
    (setq tfl- (instance ros::transform-listener :init))
    (setq object-boxes-in-bin- (make-hash-table))
    (setq bin-movable-regions- (make-hash-table))
    (setq object-boxes-in-bin- (make-hash-table))
    (setq object-coords-in-bin- (make-hash-table)))
  (:tf-pose->coords
    (frame_id pose)
    (let ((base-name (send (send (send self :robot) :base_lk) :name)))
      (send (send tfl- :lookup-transform base-name frame_id (ros::time 0))
            :transform (ros::tf-pose->coords pose))))
  (:bbox->cube
    (bbox)
    (let* ((dims (ros::tf-point->pos (send bbox :dimensions)))
           (bx (make-cube (elt dims 0) (elt dims 1) (elt dims 2))))
      (send bx :newcoords
            (send self :tf-pose->coords
                  (send bbox :header :frame_id)
                  (send bbox :pose)))
      bx))
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
  (:set-movable-region-for-bin
    (&key (offset (list 0 0 0)))
    (let (cube)
        (progn
          (dolist (bin (list :a :b :c))
               (setq cube (send self :bbox->cube (gethash bin bin-boxes-)))
               (sethash bin bin-movable-regions-
                        (send self :cube->movable-region cube :offset offset))))))
  (:need-to-wait-opposite-arm
    (arm state)
    (let (opposite-state)
      (setq opposite-state
            (str2symbol (ros::get-param
                          (format nil "~a_hand/state"
                                  (arm2str (get-opposite-arm arm))))))
      (if (eq state :wait_for_opposite_arm_start_picking)
        (null (or (eq opposite-state :recognize_objects_in_bin)
                  (eq opposite-state :wait_for_user_input)))
        (or (eq opposite-state :recognize_objects_in_bin)
            (eq opposite-state :pick_object)
            (eq opposite-state :verify_object)
            (eq opposite-state :set_target_cardboard)
            (eq opposite-state :return_object)))))
  (:get-work-orders
    (arm)
    (let (msg)
      (setq msg
        (one-shot-subscribe
          (format nil "/strategic_work_order/~a_hand" (arm2str arm))
          jsk_arc2017_common::WorkOrderArray))
      (send msg :orders)))
  (:get-next-work-order
    (arm current-order)
    (let ((orders (send self :get-work-orders arm)))
      (when (eq (length orders) 0)
        (ros::ros-error "[:get-next-work-order] There is no order")
        (return-from :get-next-work-order nil))
      (when (null current-order) (return-from :get-next-work-order (elt orders 0)))
      (dotimes (i (- (length orders) 1))
        (when (string= (send (elt orders i) :item) (send current-order :item))
          (return-from :get-next-work-order (elt orders (+ i 1)))))))
  (:get-certain-work-order
    (arm bin)
    (let ((orders (send self :get-work-orders arm)))
      (when (eq (length orders) 0)
        (ros::ros-error "[:get-certain-work-order] There is no order")
        (return-from :get-certain-work-order nil))
      (when (null bin) (return-from :get-certain-work-order (elt orders 0)))
      (dotimes (i (length orders))
        (when (string= (send (elt orders i) :bin)
                       (string-upcase (symbol2str bin)))
          (return-from :get-certain-work-order (elt orders i))))))
  (:check-bin-exist (bin) (if (gethash bin bin-boxes-) t nil))
  (:recognize-bin-boxes
    (&key (stamp (ros::time-now)))
    (let ((box-topic (format nil "publish_bin_boxes/output"))
          box-msg bin-list)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[~a] [:recognize-bin-boxes] recognize bin boxes" (ros::get-name))
          (setq box-list (send box-msg :boxes))
          (setq bin-list (list :a :b :c))
          (dolist (bin bin-list)
            (setf (gethash bin bin-boxes-) (car box-list))
            (setq box-list (cdr box-list))))
        (ros::ros-fatal "[:recognize-bin-boxes] cannot recognize bin boxes"))))
  (:visualize-bins ()
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
          (pushback bin-model bins)))
      bins))
  (:recognize-objects-in-bin
    (arm &key (stamp (ros::time-now)) (timeout 10))
    (let ((box-topic (format nil "~a_hand_camera/cluster_indices_decomposer_target/boxes" (arm2str arm)))
          box-msg
          (com-topic (format nil "~a_hand_camera/cluster_indices_decomposer_target/centroid_pose_array"
                        (arm2str arm)))
          com-msg obj-box obj-coords is-recognized)
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
         (ros::ros-info "[:recognize-objects-in-bin] arm: ~a get cpi msg" arm)
         (setq obj-box (send box-msg :boxes))
         (setq obj-coords
               (mapcar #'(lambda (obj-pose)
                           (send self :tf-pose->coords
                                 (send com-msg :header :frame_id) obj-pose))
                       (send com-msg :poses)))
         (if (and (> (length obj-box) 0) (> (length obj-coords) 0))
           (progn
             (sethash arm object-boxes-in-bin- obj-box)
             (sethash arm object-coords-in-bin- obj-coords)
             (setq is-recognized t))
           (progn
             (ros::ros-error "[:recognize-objects-in-bin] arm: ~a obj-box length ~a" arm (length obj-box))
             (ros::ros-error "[:recognize-objects-in-bin] arm: ~a obj-coords length ~a" arm (length obj-coords))
             (setq is-recognized nil))))
        (t
         (ros::ros-error "[:recognize-objects-in-bin] arm: ~a failed to get cpi msg" arm)
         (setq is-recognized nil)))
      is-recognized))
  (:pick-object-in-bin
    (arm bin &key (n-trial 1) (n-trial-same-pos 1) (do-stop-grasp nil))
    (let (graspingp object-index obj-pos avs)
      (if (or
          (null (gethash arm object-boxes-in-bin-))
          (null (gethash arm object-coords-in-bin-)))
        (return-from :pick-object-in-bin nil) t)
      (setq object-index (random (length (gethash arm object-boxes-in-bin-))))
      (setq obj-pos (send self :get-object-position arm bin :object-index object-index))
      (send self :angle-vector
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos (v+ obj-pos #f(0 0 150))
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            :fast (get-arm-controller arm) 0 :scale 5.0)
      (send self :wait-interpolation)
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (unless graspingp
            (setq graspingp
                  (send self :try-to-pick-object-in-bin arm obj-pos
                        :offset (float-vector 0 0 (- (* i -50) 30))))
            (pushback (send *baxter* :angle-vector) avs))))
      (when do-stop-grasp (unless graspingp (send self :stop-grasp arm)))
      (send self :angle-vector-sequence (reverse avs)
            :fast (get-arm-controller arm) 0 :scale 5.0)
      (send self :wait-interpolation)
      (send self :gripper-servo-on arm)
      graspingp))
  (:get-object-position
    (arm bin &key (object-index 0))
    (let (avs obj-box bin-box movable-region obj-coords obj-pos obj-box-z-length)
      ;; validate
      (unless
        (setq obj-box (elt (gethash arm object-boxes-in-bin-) object-index))
        (ros::ros-warn "[:get-object-position] No bbox  is found: ~a, ~a" arm bin)
        (return-from :get-object-position nil))
      ;; with Center of Mass
      (unless
        (setq obj-coords (elt (gethash arm object-coords-in-bin-) object-index))
        (ros::ros-warn "[:get-object-position] No com is found: ~a, ~a" arm bin)
        (return-from :get-object-position nil))
      (unless
        (setq bin-box (gethash bin bin-boxes-))
        (ros::ros-error "[:get-object-position] No data about order bin box. Call :recognize-bin-box first.")
        (return-from :get-object-position nil))
      (unless
        (setq movable-region (gethash bin bin-movable-regions-))
        (ros::ros-error "[:get-object-position] No data about movable region for order bin box. Call :set-movable-region-for-bin first.")
        (return-from :get-object-position nil))
      ;; grasp object
      (ros::ros-info "[:get-object-position] arm:~a approach to the object" arm)
      ;; gripper is straight
      (send self :gripper-servo-on arm)
      (setq obj-pos (send obj-coords :worldpos))
      (dolist (i (list 0 1))
        (cond ((> (elt (elt movable-region i) 0) (aref obj-pos i))
               (ros::ros-info "[:get-object-position] object is out of movable region. ~a > ~a < ~a"
                              (elt (elt movable-region i) 1)
                              (elt (send obj-coords :pos) i)
                              (elt (elt movable-region i) 0))
               (setf (aref obj-pos i) (elt (elt movable-region i) 0)))
              ((> (aref obj-pos i) (elt (elt movable-region i) 1))
               (ros::ros-info "[:get-object-position] object is out of movable region. ~a < ~a > ~a"
                              (elt (elt movable-region i) 1)
                              (elt (send obj-coords :pos) i)
                              (elt (elt movable-region i) 0))
               (setf (aref obj-pos i) (elt (elt movable-region i) 1)))
              (t nil)))
      (setq obj-box-z-length (z-of-cube (send self :bbox->cube obj-box)))
      (setq obj-pos (v+ obj-pos (float-vector 0 0 (/ obj-box-z-length 2))))
      obj-pos))
  (:try-to-pick-object-in-bin
    (arm obj-pos &key (offset #f(0 0 0)))
    (let (graspingp)
      (send self :angle-vector
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos (v+ obj-pos offset)
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            3000 (get-arm-controller arm) 0)
      (send self :wait-interpolation)
      ;; start the vacuum gripper after approaching to the object
      (ros::ros-info "[:try-to-pick-object-in-bin] arm:~a start vacuum gripper" arm)
      (send self :start-grasp arm)
      (unix::sleep 1)
      (send self :angle-vector
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos obj-pos
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            3000 (get-arm-controller arm) 0)
      (send self :wait-interpolation)
      (setq graspingp (send self :graspingp arm))
      (ros::ros-info "[:try-to-pick-object-in-bin] arm:~a graspingp: ~a" arm graspingp)
      (unless graspingp
        (ros::ros-info "[:try-to-pick-object-in-bin] arm:~a again approach to the object" arm)
        (let ((temp-av (send *baxter* :angle-vector)))
          ;; only if robot can solve IK
          (if (send *baxter* arm :move-end-pos #f(0 0 -50) :local)
            (send self :angle-vector (send *baxter* :angle-vector)
                  3000 (get-arm-controller arm) 0))
          (send self :wait-interpolation)
          (send self :angle-vector (send *baxter* :angle-vector temp-av)
                  3000 (get-arm-controller arm) 0) ;; revert baxter
          (send self :wait-interpolation)))
      ;; lift object
      (ros::ros-info "[:try-to-pick-object-in-bin] arm:~a lift the object" arm)
      (send self :gripper-servo-off arm)
      (send self :angle-vector (send *baxter* arm :move-end-pos #f(0 0 200) :world)
            3000 (get-arm-controller arm) 0)
      (send self :wait-interpolation)
      (unix::sleep 1)  ;; wait for arm to follow
      (setq graspingp (send self :graspingp arm))
      (ros::ros-info "[:try-to-pick-object-in-bin] arm:~a graspingp: ~a" arm graspingp)
    graspingp))
  (:ik->bin-center
    (arm bin &key (offset #f(0 0 0)) (coords-pitch 0) (rotation-axis t) (use-gripper nil))
    (let (bin-box bin-coords)
      (setq bin-box (gethash bin bin-boxes-))
      (setq bin-coords (send self :tf-pose->coords
                             (send (send bin-box :header) :frame_id)
                             (send bin-box :pose)))
      (send bin-coords :translate
            (v+ (float-vector
                  0.0 0.0 (/ (m->mm (send bin-box :dimensions :z)) 2.0))
                 offset) :world)
      (send bin-coords :rotate (* pi (/ coords-pitch 180)) :y)
      (send *baxter* arm :inverse-kinematics bin-coords
            :rotation-axis rotation-axis
            :use-gripper use-gripper)))
  (:ik->bin-entrance
    (arm bin &key (offset #f(0 0 0)) (coords-pitch 0) (rotation-axis t) (use-gripper nil))
    (let (bin-box bin-coords)
      (setq bin-box (gethash bin bin-boxes-))
      (setq bin-coords (send self :tf-pose->coords
                             (send (send bin-box :header) :frame_id)
                             (send bin-box :pose)))
      (send bin-coords :translate
            (v+ (float-vector
                  (- (/ (m->mm (send bin-box :dimensions :x)) 2.0))
                  0.0 (/ (m->mm (send bin-box :dimensions :z)) 2.0))
                offset) :world)
      (send bin-coords :rotate (* pi (/ coords-pitch 180)) :y)
      (send *baxter* arm :inverse-kinematics bin-coords
            :rotation-axis rotation-axis
            :use-gripper use-gripper)))
  (:ik->cardboard-entrance
    (arm cardboard &key (offset #f(0 0 0)) (rotation-axis t) (use-gripper nil))
    (send *baxter* arm :inverse-kinematics
          (make-coords
            :pos (v+ offset
                     (cond ((eq cardboard :a) (float-vector 950 500 0))
                           ((eq cardboard :b) (float-vector 650 500 0))
                           (t (float-vector 800 -500 0))))
            :rpy (float-vector 0 0 0))
          :rotation-axis rotation-axis
          :use-gripper use-gripper))
  (:move-arm-body->bin-overlook-pose
    (arm bin &key (gripper-angle 90))
    (let (avs)
      (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
      (send *baxter* :reset-pose arm)
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (pushback
        (send self :ik->bin-entrance arm bin
              :offset #f(-100 0 300) :coords-pitch gripper-angle
              :rotation-axis :y
              :use-gripper nil) avs)
      (send self :angle-vector-sequence avs :fast (get-arm-controller arm) 0 :scale 3.0)))
  (:wait-for-user-input-to-start (arm)
    (let (can-start)
      (ros::ros-info "[:wait-for-user-input-to-start] wait for user input to start: ~a" arm)
      (ros::wait-for-service "/rviz/yes_no_button")
      (while
        (not can-start)
        (setq can-start (send (ros::service-call
                                "/rviz/yes_no_button" (instance jsk_gui_msgs::YesNoRequest)) :yes)))
      (ros::ros-info "[:wait-for-user-input-to-start] received user input: ~a" arm)))
  (:set-object-segmentation-candidates (arm candidates)
    (let ((req (instance jsk_recognition_msgs::SetLabelsRequest :init)))
      (send req :labels candidates)
      (ros::service-call
        (format nil "/~a_hand_camera/apply_context_to_label_proba/update_candidates"
                (arm2str arm))
        req)))
  (:set-arm-state-param (arm state)
    (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state)))
  (:get-bin-contents (bin)
    (ros::get-param
      (format nil "/bin_contents/~A" (string-upcase (symbol2str bin)))))
  )

(defun jsk_arc2017_baxter::arc-init (&key (ctype :default-controller) (moveit nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance jsk_2016_01_baxter_apc::baxter-moveit-environment))
      (setq mvit-rb (instance jsk_2016_01_baxter_apc::baxter-robot :init)))
    (unless (boundp '*ri*)
      (setq *ri* (instance jsk_arc2017_baxter::arc-interface :init :type ctype
                           :moveit-environment mvit-env
                           :moveit-robot mvit-rb)))
    (unless (boundp '*baxter*)
      (setq *baxter* (instance jsk_2016_01_baxter_apc::baxter-robot :init)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)))
