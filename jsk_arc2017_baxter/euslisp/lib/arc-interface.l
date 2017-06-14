;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/baxter.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/baxter-interface.l")

(ros::load-ros-manifest "jsk_arc2017_baxter")

(unless (find-package "JSK_ARC2017_BAXTER")
  (make-package "JSK_ARC2017_BAXTER"))

(defclass jsk_arc2017_baxter::arc-interface
  :super propertied-object
  :slots (bin-cubes-
          bin-movable-regions-
          object-boxes-
          object-coords-
          larm-cardboard-cube-
          rarm-cardboard-cube-
          tote-cubes-
          tote-movable-regions-
          ))

(defmethod jsk_arc2017_baxter::arc-interface
  (:init ()
    ;; initialize slots
    (setq bin-cubes- (make-hash-table))
    (setq bin-movable-regions- (make-hash-table))
    (setq object-boxes- (make-hash-table))
    (setq object-coords- (make-hash-table))
    (setq rarm-cardboard-cube- (make-cube 800 400 300))
    (send rarm-cardboard-cube- :newcoords
          (make-coords :pos (float-vector 800 -500 -100)))
    (setq larm-cardboard-cube- (make-cube 800 400 300))
    (send larm-cardboard-cube- :newcoords
          (make-coords :pos (float-vector 800 500 -100)))
    (setq tote-cubes- (make-hash-table))
    (setq tote-movable-regions- (make-hash-table)))
  (:fold-pose-back
    (&optional (arm :arms) &rest args)
    (dolist (tmp-arm (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (send *baxter* :fold-pose-back tmp-arm))
    (send* *ri* :angle-vector (send *baxter* :angle-vector) args))
  (:bbox->cube
    (bbox)
    (let* ((dims (ros::tf-point->pos (send bbox :dimensions)))
           (bx (make-cube (elt dims 0) (elt dims 1) (elt dims 2))))
      (send bx :newcoords
            (send *ri* :tf-pose->coords
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
               (setq cube (gethash bin bin-cubes-))
               (sethash bin bin-movable-regions-
                        (send self :cube->movable-region cube :offset offset))))))
  (:set-movable-region-for-tote
    (arm &key (offset (list 0 0 0)))
    (let (cube)
      (setq cube (gethash arm tote-cubes-))
      (sethash arm tote-movable-regions- (send self :cube->movable-region cube :offset offset))))
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
                       (symbol-string bin))
          (return-from :get-certain-work-order (elt orders i))))))
  (:check-bin-exist (bin) (if (gethash bin bin-cubes-) t nil))
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
            (sethash bin bin-cubes- (send self :bbox->cube (car box-list)))
            (setq box-list (cdr box-list))))
        (ros::ros-fatal "[:recognize-bin-boxes] cannot recognize bin boxes"))))
  (:recognize-tote-boxes
    (arm &key (stamp (ros::time-now)))
    (let ((box-topic (format nil "~a_hand_camera/attention_clipper_tote/output/box_array"
                             (arm2str arm)))
                     box-msg bbox)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[:recognize-tote-boxes] arm: ~a Recognize tote bbox" (arm2str arm))
          (setq bbox (car (send box-msg :boxes)))
          (sethash arm tote-cubes- (send self :bbox->cube bbox)))
        (ros::ros-fatal "[:recognize-tote-boxes] arm: ~a Cannot find tote bbox" (arm2str arm)))))
  (:visualize-bins ()
    (let (bins)
      (dolist (box-inside (send bin-cubes- :list-values))
        (let ((bin-outside (make-cube (+ (x-of-cube box-inside) 30)
                                      (+ (y-of-cube box-inside) 30)
                                      (z-of-cube box-inside)))
              (bin-model))
          (send bin-outside :newcoords (send bin-inside :copy-worldcoords))
          (send bin-outside :translate (float-vector 0 0 15) :world)
          (setq bin-model (body- bin-outside bin-inside))
          (send bin-model :set-color :blue 0.5)
          (pushback bin-model bins)))
      bins))
  (:recognize-target-object
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
      (while (and (not (and box-msg com-msg))
                  (> (+ (send stamp :to-sec) timeout) (send (ros::time-now) :to-sec)))
             (unix::usleep (* 50 1000))
             (ros::spin-once))
      (ros::unsubscribe box-topic)
      (ros::unsubscribe com-topic)
      (cond
        ((and box-msg com-msg)
         (ros::ros-info "[:recognize-target-object] arm: ~a get cpi msg" arm)
         (setq obj-box (send box-msg :boxes))
         (setq obj-coords
               (mapcar #'(lambda (obj-pose)
                           (send *ri* :tf-pose->coords
                                 (send com-msg :header :frame_id) obj-pose))
                       (send com-msg :poses)))
         (if (and (> (length obj-box) 0) (> (length obj-coords) 0))
           (progn
             (sethash arm object-boxes- obj-box)
             (sethash arm object-coords- obj-coords)
             (setq is-recognized t))
           (progn
             (ros::ros-error "[:recognize-target-object] arm: ~a obj-box length ~a" arm (length obj-box))
             (ros::ros-error "[:recognize-target-object] arm: ~a obj-coords length ~a" arm (length obj-coords))
             (setq is-recognized nil))))
        (t
         (ros::ros-error "[:recognize-target-object] arm: ~a failed to get cpi msg" arm)
         (setq is-recognized nil)))
      is-recognized))
  (:pick-object-in-bin
    (arm &rest args)
    (let (graspingp movable-region)
      (setq movable-region (gethash bin bin-movable-regions-))
      (unless movable-region
        (ros::ros-error "[:pick-object-in-bin] No movable region for bin. Call :set-movable-region-for-bin first.")
        (return-from :pick-object-in-bin nil))
      (setq graspingp (send* self :pick-object-with-movable-region arm movable-region args))
      graspingp))
  (:pick-object-in-tote
    (arm &rest args)
    (let (graspingp movable-region)
      (setq movable-region (gethash arm tote-movable-regions-))
      (unless movable-region
        (ros::ros-error "[:pick-object-in-tote] No movable region for tote. Call :set-movable-region-for-tote first.")
        (return-from :pick-object-in-tote nil))
      (setq graspingp (send* self :pick-object-with-movable-region arm movable-region args))
      graspingp))
  (:pick-object-with-movable-region
    (arm movable-region &key (n-trial 1) (n-trial-same-pos 1) (do-stop-grasp nil))
    (let (graspingp avs object-index)
      ;; TODO: object-index is set randomly
      (setq object-index (random (length (gethash arm object-boxes-))))
      (setq obj-pos
            (send self :get-object-position arm movable-region :object-index object-index))
      (ros::ros-info "[:pick-object-with-movable-region] arm:~a approach to the object" arm)
      (send *ri* :gripper-servo-on arm)
      (send *ri* :angle-vector
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos (v+ obj-pos #f(0 0 150))
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
      (send *ri* :wait-interpolation)
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (unless graspingp
            (setq graspingp
                  (send self :try-to-pick-object arm obj-pos
                        :offset (float-vector 0 0 (- (* i -50) 30))))
            (pushback (send *baxter* :angle-vector) avs))))
      (when do-stop-grasp (unless graspingp (send *ri* :stop-grasp arm)))
      (send *ri* :angle-vector-sequence (reverse avs)
            :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
      (send *ri* :wait-interpolation)
      (send *ri* :gripper-servo-on arm)
      graspingp))
  (:get-object-position
    (arm movable-region &key (object-index 0))
    (let (obj-box obj-coords obj-pos obj-box-z-length)
      (setq obj-box (elt (gethash arm object-boxes-) object-index))
      (setq obj-coords (elt (gethash arm object-coords-) object-index))
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
  (:try-to-pick-object
    (arm obj-pos &key (offset #f(0 0 0)))
    (let (graspingp)
      (send *ri* :angle-vector-raw
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos (v+ obj-pos offset)
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      ;; start the vacuum gripper after approaching to the object
      (ros::ros-info "[:try-to-pick-object] arm:~a start vacuum gripper" arm)
      (send *ri* :start-grasp arm)
      (unix::sleep 1)
      (send *ri* :angle-vector-raw
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos obj-pos
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      (setq graspingp (send *ri* :graspingp arm))
      (ros::ros-info "[:try-to-pick-object] arm:~a graspingp: ~a" arm graspingp)
      (unless graspingp
        (ros::ros-info "[:try-to-pick-object] arm:~a again approach to the object" arm)
        (let ((temp-av (send *baxter* :angle-vector)))
          ;; only if robot can solve IK
          (if (send *baxter* arm :move-end-pos #f(0 0 -50) :local)
            (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
                  3000 (send *ri* :get-arm-controller arm) 0))
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector-raw (send *baxter* :angle-vector temp-av)
                  3000 (send *ri* :get-arm-controller arm) 0) ;; revert baxter
          (send *ri* :wait-interpolation)))
      ;; lift object
      (ros::ros-info "[:try-to-pick-object] arm:~a lift the object" arm)
      (send *ri* :gripper-servo-off arm)
      (send *ri* :angle-vector-raw (send *baxter* arm :move-end-pos #f(0 0 200) :world)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      (unix::sleep 1)  ;; wait for arm to follow
      (setq graspingp (send *ri* :graspingp arm))
      (ros::ros-info "[:try-to-pick-object] arm:~a graspingp: ~a" arm graspingp)
    graspingp))
  (:ik->bin-center
    (arm bin &key (offset #f(0 0 0)) (rpy #f(0 0 0)) (rotation-axis t) (use-gripper nil))
    (let (bin-cube bin-coords)
      (setq bin-cube (gethash bin bin-cubes-))
      (setq bin-coords (send bin-cube :copy-worldcoords))
      (send bin-coords :rotate (aref rpy 0) :z)
      (send bin-coords :rotate (aref rpy 1) :y)
      (send bin-coords :rotate (aref rpy 2) :x)
      (send bin-coords :translate
            (v+ (float-vector
                  0.0 0.0 (/ (z-of-cube bin-cube) 2.0))
                offset) :local)
      (send *baxter* arm :inverse-kinematics bin-coords
            :rotation-axis rotation-axis
            :use-gripper use-gripper)))
  (:ik->cardboard-center
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
  (:ik->tote-center
    (arm &key (offset #f(0 0 0)) (rpy #f(0 0 0)) (rotation-axis t) (use-gripper nil))
    (let (tote-cube tote-coords)
      (setq tote-cube (gethash arm tote-cubes-))
      (setq tote-coords (send tote-cube :copy-worldcoords))
      (send tote-coords :rotate (aref rpy 0) :z)
      (send tote-coords :rotate (aref rpy 1) :y)
      (send tote-coords :rotate (aref rpy 2) :x)
      (send tote-coords :translate
            (v+ (float-vector
                  0.0 0.0 (/ (z-of-cube tote-cube) 2.0))
                offset) :local)
      (send *baxter* arm :inverse-kinematics tote-coords
            :rotation-axis rotation-axis
            :use-gripper use-gripper)))
  (:move-arm-body->bin-overlook-pose
    (arm bin &key (gripper-angle 90))
    (let (avs offset rpy
              (offset-y (if (eq arm :larm) 180 -180)))
      (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
      (send *baxter* :reset-pose arm)
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (setq offset
            (cond ((eq bin :a) (float-vector 0 offset-y 250))
                  ((eq bin :b) (float-vector 0 offset-y 200))
                  ((eq bin :c) (float-vector 0 offset-y 200))))
      (setq rpy
            (float-vector 0 pi/2 (if (eq arm :larm) pi/2 -pi/2)))
      (pushback
        (send self :ik->bin-center arm bin
              :offset offset :rpy rpy :use-gripper nil)
        avs)
      (send *ri* :angle-vector-sequence avs :fast
            (send *ri* :get-arm-controller arm) 0 :scale 3.0)))
  (:move-arm-body->tote-overlook-pose
    (arm &key (gripper-angle 90))
    (let (avs offset rpy
              (offset-y (if (eq arm :larm) 180 -180)))
      (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
      (send *baxter* :reset-pose arm)
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (setq offset (float-vector 0 offset-y 250))
      (setq rpy (float-vector 0 pi/2 (if (eq arm :larm) pi/2 -pi/2)))
      (pushback
        (send self :ik->tote-center arm
              :offset offset :rpy rpy :use-gripper nil)
        avs)
      (send *ri* :angle-vector-sequence avs :fast
            (send *ri* :get-arm-controller arm) 0 :scale 3.0)))
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
      (format nil "/bin_contents/~A" (symbol-string bin))))
  (:add-bin-scene (bin)
    (when (null *co*)
      (ros::ros-error "[:add-bin-scene] *co* is nil. Skipped")
      (return-from :add-bin-scene nil))
    (let ((cube (gethash bin bin-cubes-))
          (base-name (send (send *baxter* :base_lk) :name)))
      (send *co* :add-object cube :frame-id base-name
            :relative-pose (send cube :copy-worldcoords)
            :object-id (format nil "bin_~A" (symbol-string bin)))))
  (:delete-bin-scene (bin)
    (when (null *co*)
      (ros::ros-error "[:delete-bin-scene] *co* is nil. Skipped")
      (return-from :delete-bin-scene nil))
    (send *co* :delete-object (gethash bin bin-cubes-)))
  (:add-shelf-scene ()
    (dolist (bin (list :a :b :c))
      (send self :add-bin-scene bin)))
  (:delete-shelf-scene ()
    (dolist (bin (list :a :b :c))
      (send self :delete-bin-scene bin)))
  (:add-cardboard-scene (arm)
    (when (null *co*)
      (ros::ros-error "[:add-cardboard-scene] *co* is nil. Skipped")
      (return-from :add-cardboard-scene nil))
    (let ((cardboard-cube (if (eq arm :larm) larm-cardboard-cube- rarm-cardboard-cube-))
          (base-name (send (send *baxter* :base_lk) :name)))
      (send *co* :add-object cardboard-cube :frame-id base-name
            :relative-pose (send cardboard-cube :copy-worldcoords)
            :object-id (format nil "~a_cardboard" (arm2str arm)))))
  (:delete-cardboard-scene (arm)
    (when (null *co*)
      (ros::ros-error "[:delete-cardboard-scene] *co* is nil. Skipped")
      (return-from :delete-cardboard-scene nil))
    (let ((cardboard-cube (if (eq arm :larm) larm-cardboard-cube- rarm-cardboard-cube-)))
      (send *co* :delete-object cardboard-cube)))
  (:add-tote-scene (arm)
    (when (null *co*)
      (ros::ros-error "[:add-tote-scene] *co* is nil. Skipped")
      (return-from :add-tote-scene nil))
    (let ((cube (gethash arm tote-cubes-))
          (base-name (send (send *baxter* :base_lk) :name)))
      (send *co* :add-object cube :frame-id base-name
            :relative-pose (send cube :copy-worldcoords)
            :object-id (format nil "~A_tote" (arm2str arm)))))
  (:delete-tote-scene (arm)
    (when (null *co*)
      (ros::ros-error "[:delete-tote-scene] *co* is nil. Skipped")
      (return-from :delete-tote-scene nil))
    (send *co* :delete-object (gethash arm tote-cubes-)))
  (:wipe-all-scene ()
    (when (null *co*)
      (ros::ros-error "[:wipe-all-scene] *co* is nil. Skipped")
      (return-from :wipe-all-scene nil))
    (send *co* :wipe-all))
  (:spin-off-by-wrist
    (arm &key (times 10))
    (send *ri* :angle-vector-sequence-raw
          (send *baxter* :spin-off-by-wrist arm :times times)
          :fast (send *ri* :get-arm-controller arm) 0))
  (:send-av
    (&optional (tm 3000) (ctype nil))
    (send *ri* :angle-vector (send *baxter* :angle-vector) tm ctype))
  (:check-can-start (arm start-state wait-state)
    (let ((service-name
           (format nil "/state_server/~a_hand/check_can_start" (arm2str arm)))
          req can-start)
      (ros::wait-for-service service-name)
      (setq req (instance jsk_arc2017_baxter::CheckCanStartRequest :init))
      (send req :start_state (symbol2str start-state))
      (send req :wait_state (symbol2str wait-state))
      (setq can-start
            (send (ros::service-call service-name req) :can_start))
      can-start))
  (:update-state (arm state)
    (let ((service-name
           (format nil "/state_server/~a_hand/update_state" (arm2str arm)))
          req updated)
      (ros::wait-for-service service-name)
      (setq req (instance jsk_arc2017_baxter::UpdateStateRequest :init))
      (send req :state (symbol2str state))
      (setq updated (send (ros::service-call service-name req) :updated))
      updated))
  (:get-state (arm)
    (let ((service-name
           (format nil "/state_server/~a_hand/get_state" (arm2str arm)))
          req state)
      (ros::wait-for-service service-name)
      (setq req (instance jsk_arc2017_baxter::GetStateRequest :init))
      (setq state (send (ros::service-call service-name req) :state))
      state)))

(defun jsk_arc2017_baxter::arc-init (&key (ctype :default-controller) (moveit nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance jsk_2016_01_baxter_apc::baxter-moveit-environment))
      (setq mvit-rb (instance jsk_arc2017_baxter::baxter-robot :init)))
    (unless (boundp '*ri*)
      (setq *ri* (instance jsk_arc2017_baxter::baxter-interface :init :type ctype
                           :moveit-environment mvit-env
                           :moveit-robot mvit-rb)))
    (unless (boundp '*baxter*)
      (setq *baxter* (instance jsk_arc2017_baxter::baxter-robot :init)))
    (unless (boundp '*co*)
      (setq *co* (when moveit (instance collision-object-publisher :init))))
    (unless (boundp '*ti*)
      (setq *ti* (instance jsk_arc2017_baxter::arc-interface :init)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)))
