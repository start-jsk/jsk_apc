;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/baxter-interface.l")

(ros::load-ros-manifest "jsk_arc2017_baxter")

(unless (find-package "JSK_ARC2017_BAXTER")
  (make-package "JSK_ARC2017_BAXTER"))

(defclass jsk_arc2017_baxter::arc-interface
  :super propertied-object
  :slots (bbox-volume-threshold-
          bin-cubes-
          bin-movable-regions-
          object-boxes-
          object-coords-
          cardboard-cubes-
          tote-cubes-
          tote-movable-regions-
          ))

(defmethod jsk_arc2017_baxter::arc-interface
  (:init ()
    ;; initialize slots
    (setq bbox-volume-threshold- 0.01) ;; [m^3]
    (setq bin-cubes- (make-hash-table))
    (setq bin-movable-regions- (make-hash-table))
    (setq object-boxes- (make-hash-table))
    (setq object-coords- (make-hash-table))
    (setq cardboard-cubes- (make-hash-table))
    (setq upper-cardboard-cube (make-cube 400 800 200))
    (send upper-cardboard-cube :newcoords
          (make-coords :pos (float-vector 1200 0 550)
                       :rpy (float-vector 0 (/ -pi/2 2) 0)))
    (setq lower-cardboard-cube (make-cube 400 800 200))
    (send lower-cardboard-cube :newcoords
          (make-coords :pos (float-vector 1200 0 165)
                       :rpy (float-vector 0 (/ -pi/2 2) 0)))
    (sethash :upper cardboard-cubes- upper-cardboard-cube)
    (sethash :lower cardboard-cubes- lower-cardboard-cube)
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
    (&key (stamp (ros::time-now)))
    (let ((box-topic (format nil "publish_tote_boxes/output"))
                     box-msg box-list)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[~a] [:recognize-tote-boxes] recognize tote bbox" (ros::get-name))
          (setq bbox-list (send box-msg :boxes))
          (dolist (arm '(:larm :rarm))
            (sethash arm tote-cubes- (send self :bbox->cube (pop bbox-list)))))
        (ros::ros-fatal "[:recognize-tote-boxes] cannot recognize tote bbox"))))
  (:visualize-bins ()
    (ros::ros-warn ":visualize-bins was deprecated. Please use :visualize-boxes.")
    (send self :visualize-boxes :box-type :bin))
  (:visualize-boxes (&key (box-type :bin))
    "Visualize boxes
     Arguments
     ---------
     box-type: symbol
       :bin, :tote or :cardboard"
    (let (boxes box-color box-cubes)
      (cond
        ((eq box-type :bin) (setq box-cubes bin-cubes- box-color :blue))
        ((eq box-type :tote) (setq box-cubes tote-cubes- box-color :red))
        ((eq box-type :cardboard) (ros::ros-error "Not implemented box-type: :cardboard"))
        (t (ros::ros-error "Unsupported box-type ~a" box-type))
        )
      (dolist (box-inside (send box-cubes :list-values))
        (let ((box-outside (make-cube (+ (x-of-cube box-inside) 30)
                                      (+ (y-of-cube box-inside) 30)
                                      (z-of-cube box-inside)))
              (box-model))
          (send box-outside :newcoords (send box-inside :copy-worldcoords))
          (send box-outside :translate (float-vector 0 0 15) :world)
          (setq box-model (body- box-outside box-inside))
          (send box-model :set-color box-color 0.5)
          (pushback box-model boxes)))
      boxes))
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
  (:get-largest-object-index
    (arm &key (stamp (ros::time-now)))
    (let ((box-topic (format nil "~a_hand_camera/cluster_indices_decomposer_label/boxes" (arm2str arm)))
          box-msg)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (dolist (box (send box-msg :boxes))
        (when box
          (let (volume)
            (setq volume (* (send box :dimensions :x)
                            (send box :dimensions :y)
                            (send box :dimensions :z)))
            (if (> bbox-volume-threshold- volume)
              (return-from :get-largest-object-index
                           (+ (send box :label) 1))))))
      nil))
  (:move-fingers-pinch->init
    (arm)
    ;; Move fingers to initial pose
    (send *ri* :stop-grasp arm :pinch)
    (send *ri* :move-hand arm
          (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000))
  (:pick-object-in-bin
    (arm bin &rest args)
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
    (arm movable-region &key (n-trial 1) (n-trial-same-pos 1)
         (do-stop-grasp nil) (grasp-style :suction))
    (let (graspingp avs object-index obj-pos obj-cube pinch-yaw)
      ;; TODO: object-index is set randomly
      (setq object-index (random (length (gethash arm object-boxes-))))
      (setq obj-pos
            (send self :get-object-position arm movable-region :object-index object-index))
      (setq obj-cube (send self :bbox->cube (elt (gethash arm object-boxes-) object-index)))
      (if (> (x-of-cube obj-cube) (y-of-cube obj-cube)) (setq pinch-yaw pi/2) (setq pinch-yaw 0))
      (ros::ros-info "[:pick-object-with-movable-region] arm:~a approach to the object" arm)
      (send *ri* :gripper-servo-on arm)
      ;; Setup arm for picking
      (if (eq arm :larm)
        (progn
          (send *ri* :angle-vector-raw
                (send *baxter* arm :inverse-kinematics
                      (make-coords :pos (v+ obj-pos #f(0 0 150))
                                   :rpy #f(0 0 0))
                      :use-gripper t
                      :rotation-axis :z)
                :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
          (send *ri* :wait-interpolation))
        (let (prev-av next-av gripper-x)
          (setq prev-av (send *baxter* :angle-vector))
          (if (eq grasp-style :suction)
            (setq next-av
                  (send *baxter* arm :inverse-kinematics
                        (make-coords :pos (v+ obj-pos #f(0 0 150))
                                     :rpy #f(0 0 0))
                        :use-gripper t
                        :rotation-axis :z
                        :stop 200
                        :additional-check
                        #'(lambda ()
                            (and
                              (< (send *baxter* arm :gripper-x :joint-angle) 110)
                              (< -30 (send *baxter* arm :gripper-p :joint-angle) 30)))))
            (progn
              (send *baxter* :rotate-gripper :rarm 0 :relative nil)
              (send *baxter* :slide-gripper :rarm 60 :relative nil)
              (setq next-av
                    (send *baxter* arm :inverse-kinematics
                          (make-coords :pos (v+ obj-pos #f(0 0 200))
                                       :rpy (float-vector pinch-yaw 0 0))
                          :move-palm-end t
                          :rotation-axis t))))
          (setq gripper-x (send *baxter* arm :gripper-x :joint-angle))
          (send *baxter* :angle-vector prev-av)
          ;; First, move prismatic joint to target position
          (send *ri* :angle-vector-raw
                (send *baxter* :slide-gripper arm gripper-x :relative nil)
                :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
          (send *ri* :wait-interpolation)
          ;; Fold fingers to avoid collision
          (send *ri* :move-hand arm
                (send *baxter* :hand-grasp-pre-pose arm :cylindrical) 1000)
          (send *ri* :move-hand arm
                (send *baxter* :hand-grasp-pose arm :cylindrical :angle 100) 1000)
          ;; Move whole arm to target pose
          (send *ri* :angle-vector-raw
                (send *baxter* :angle-vector next-av)
                :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
          (send *ri* :wait-interpolation)))
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (unless graspingp
            (setq graspingp
                  (send self :try-to-pick-object arm obj-pos
                        :offset (float-vector 0 0 (- (* i -50) 30))
                        :grasp-style grasp-style
                        :pinch-yaw pinch-yaw)))))
      (when do-stop-grasp
        (unless graspingp
          (send *ri* :stop-grasp arm)
          (if (eq grasp-style :pinch)
            (send self :move-fingers-pinch->init arm))))
      (if (and (eq grasp-style :suction) (eq arm :rarm))
        ;; Move underactuated fingers to initial pose
        (send *ri* :move-hand arm
              (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000))
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
    (arm obj-pos &key (offset #f(0 0 0)) (grasp-style :suction)
         (pinch-yaw (if (eq grasp-style :pinch) 0)))
      (if (eq grasp-style :suction)
        (send self :try-to-suction-object arm obj-pos :offset offset)
        (send self :try-to-pinch-object arm obj-pos pinch-yaw :offset offset)))
  (:try-to-suction-object
    (arm obj-pos &key (offset #f(0 0 0)))
    (if (eq arm :larm)
      (send self :try-to-suction-object-with-gripper-v4 arm obj-pos :offset offset)
      (send self :try-to-suction-object-with-gripper-v6 arm obj-pos :offset offset)))
  (:try-to-suction-object-with-gripper-v4
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
      (ros::ros-info "[:try-to-suction-object-with-gripper-v4] arm:~a start vacuum gripper" arm)
      (send *ri* :start-grasp arm)
      (unix::sleep 1)
      (send *ri* :angle-vector-raw
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos obj-pos
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation-until-grasp arm)
      (setq graspingp (send *ri* :graspingp arm))
      (ros::ros-info "[:try-to-suction-object-with-gripper-v4] arm:~a graspingp: ~a" arm graspingp)
      (unless graspingp
        (ros::ros-info "[:try-to-suction-object-with-gripper-v4] arm:~a again approach to the object" arm)
        (let ((temp-av (send *baxter* :angle-vector)))
          ;; only if robot can solve IK
          (if (send *baxter* arm :move-end-pos #f(0 0 -50) :local)
            (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
                  3000 (send *ri* :get-arm-controller arm) 0))
          (send *ri* :wait-interpolation-until-grasp arm)
          (send *ri* :angle-vector-raw (send *baxter* :angle-vector temp-av)
                  3000 (send *ri* :get-arm-controller arm) 0) ;; revert baxter
          (send *ri* :wait-interpolation-until-grasp arm)))
      ;; lift object
      (ros::ros-info "[:try-to-suction-object-with-gripper-v4] arm:~a lift the object" arm)
      (send *ri* :gripper-servo-off arm)
      (send *ri* :angle-vector-raw (send *baxter* arm :move-end-pos #f(0 0 200) :world)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      (unix::sleep 1)  ;; wait for arm to follow
      (setq graspingp (send *ri* :graspingp arm))
      (ros::ros-info "[:try-to-suction-object-with-gripper-v4] arm:~a graspingp: ~a" arm graspingp)
      graspingp))
  (:try-to-suction-object-with-gripper-v6
    (arm obj-pos &key (offset #f(0 0 0)))
    (let (graspingp)
      ;; Fold fingers again
      (send *ri* :move-hand arm
            (send *baxter* :hand-grasp-pose arm :cylindrical :angle 100) 1000)
      ;; start the vacuum gripper before approaching to the object
      (ros::ros-info "[:try-to-suction-object-with-gripper-v6] arm:~a start vacuum gripper" arm)
      (send *ri* :start-grasp arm)
      (send *ri* :angle-vector-raw
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos (v+ obj-pos offset)
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z
                  :stop 200
                  :additional-check
                  #'(lambda ()
                      (and
                        (< (send *baxter* arm :gripper-x :joint-angle) 110)
                        (< -30 (send *baxter* arm :gripper-p :joint-angle) 30))))
            3000 (send *ri* :get-arm-controller arm) 0)
      ;; Wait until grasp or push object to prevent tooth flying
      (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)
      (send *ri* :angle-vector-raw
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos obj-pos
                               :rpy #f(0 0 0))
                  :use-gripper t
                  :rotation-axis :z)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)
      (setq graspingp (send *ri* :graspingp arm))
      (ros::ros-info "[:try-to-suction-object-with-gripper-v6] arm:~a graspingp: ~a" arm graspingp)
      (unless graspingp
        (ros::ros-info "[:try-to-suction-object-with-gripper-v6] arm:~a again approach to the object" arm)
        (let ((temp-av (send *baxter* :angle-vector)))
          ;; only if robot can solve IK
          (if (send *baxter* arm :move-end-pos #f(0 0 -100) :local)
            (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
                  3000 (send *ri* :get-arm-controller arm) 0))
          (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)
          (send *ri* :angle-vector-raw (send *baxter* :angle-vector temp-av)
                3000 (send *ri* :get-arm-controller arm) 0) ;; revert baxter
          (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)))
      ;; Open fingers in bin
      (send *ri* :move-hand arm
            (send *baxter* :hand-grasp-pre-pose arm :cylindrical) 1000)
      ;; lift object
      (ros::ros-info "[:try-to-suction-object-with-gripper-v6] arm:~a lift the object" arm)
      (send *ri* :gripper-servo-off arm)
      (send *ri* :angle-vector-raw (send *baxter* arm :move-end-pos #f(0 0 200) :world)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      (unix::sleep 1)  ;; wait for arm to follow
      (setq graspingp (send *ri* :graspingp arm))
      (ros::ros-info "[:try-to-suction-object-with-gripper-v6] arm:~a graspingp: ~a" arm graspingp)
      graspingp))
  (:try-to-pinch-object
    (arm obj-pos pinch-yaw &key (offset #f(0 0 0)))
    (let (graspingp)
      (ros::ros-info "[:try-to-pinch-object] arm:~a pinch-yaw: ~a" arm pinch-yaw)
      ;; Initialize finger
      (send *ri* :move-hand arm
            (send *baxter* :hand-grasp-pose arm :cylindrical :angle 40) 1000)
      ;; start the vacuum gripper before approaching to the object
      (ros::ros-info "[:try-to-pinch-object] arm:~a start vacuum gripper" arm)
      (send *ri* :start-grasp arm)
      (send *ri* :angle-vector-raw
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos (v+ obj-pos offset)
                               :rpy (float-vector pinch-yaw 0 0))
                  :move-palm-end t
                  :rotation-axis t)
            3000 (send *ri* :get-arm-controller arm) 0)
      ;; Wait until grasp or finger touch
      (send *ri* :wait-interpolation-until arm
            :grasp :finger-flexion :finger-loaded :prismatic-loaded)
      (send *ri* :angle-vector-raw
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos obj-pos
                               :rpy (float-vector pinch-yaw 0 0))
                  :move-palm-end t
                  :rotation-axis t)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation-until arm
            :grasp :finger-flexion :finger-loaded :prismatic-loaded)
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (send *ri* :angle-vector-raw
            (send *baxter* :slide-gripper :rarm 120 :relative nil)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (send *ri* :start-grasp arm :pinch)
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (setq graspingp (send *ri* :graspingp arm))
      (ros::ros-info "[:try-to-pinch-object] arm:~a graspingp: ~a" arm graspingp)
      ;; lift object
      (ros::ros-info "[:try-to-pinch-object] arm:~a lift the object" arm)
      (send *ri* :angle-vector-raw (send *baxter* arm :move-end-pos #f(0 0 200) :world)
            3000 (send *ri* :get-arm-controller arm :gripper nil) 0)
      (send *ri* :wait-interpolation)
      (unix::sleep 1)  ;; wait for arm to follow
      (setq graspingp (send *ri* :graspingp arm))
      (ros::ros-info "[:try-to-pinch-object] arm:~a graspingp: ~a" arm graspingp)
      graspingp))
  (:ik->bin-center
    (arm bin &key (offset #f(0 0 0)) (rpy #f(0 0 0))
         (rotation-axis t) (use-gripper nil) (move-palm-end nil))
    (let (bin-cube bin-coords)
      (setq bin-cube (gethash bin bin-cubes-))
      (setq bin-coords (send bin-cube :copy-worldcoords))
      (send bin-coords :translate
            (v+ (float-vector
                  0.0 0.0 (/ (z-of-cube bin-cube) 2.0))
                offset) :local)
      (send bin-coords :rotate (aref rpy 0) :z)
      (send bin-coords :rotate (aref rpy 1) :y)
      (send bin-coords :rotate (aref rpy 2) :x)
      (send *baxter* arm :inverse-kinematics bin-coords
            :rotation-axis rotation-axis
            :use-gripper use-gripper
            :move-palm-end move-palm-end)))
  (:ik->cardboard-center
    (arm cardboard &key (offset #f(0 0 0))
         (rotation-axis t) (use-gripper nil) (move-palm-end nil))
    (send *baxter* arm :inverse-kinematics
          (make-coords
            :pos (v+ offset
                     (cond ((eq cardboard :a) (float-vector 1200 100 550))
                           ((eq cardboard :b) (float-vector 1200 -200 550))
                           (t (float-vector 1200 0 165))))
            :rpy (float-vector 0 (/ -pi/2 2) 0))
          :rotation-axis rotation-axis
          :use-gripper use-gripper
          :move-palm-end move-palm-end))
  (:ik->tote-center
    (arm &key (offset #f(0 0 0)) (rpy #f(0 0 0))
         (rotation-axis t) (use-gripper nil) (move-palm-end nil))
    (let (tote-cube tote-coords)
      (setq tote-cube (gethash arm tote-cubes-))
      (setq tote-coords (send tote-cube :copy-worldcoords))
      (send tote-coords :translate
            (v+ (float-vector
                  0.0 0.0 (/ (z-of-cube tote-cube) 2.0))
                offset) :local)
      (send tote-coords :rotate (aref rpy 0) :z)
      (send tote-coords :rotate (aref rpy 1) :y)
      (send tote-coords :rotate (aref rpy 2) :x)
      (send *baxter* arm :inverse-kinematics tote-coords
            :rotation-axis rotation-axis
            :use-gripper use-gripper
            :move-palm-end move-palm-end)))
  (:move-arm-body->bin-overlook-pose
    (arm bin &key (gripper-angle 90))
    (let (avs offset rpy
              (offset-y (if (eq arm :larm) 180 -180)))
      (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
      (send *baxter* :reset-pose arm)
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (setq offset (float-vector -100 offset-y 250))
      (setq rpy
            (float-vector 0 pi/2 (if (eq arm :larm) pi/2 -pi/2)))
      (pushback
        (send self :ik->bin-center arm bin
              :offset offset :rpy rpy :use-gripper nil)
        avs)
      (send *ri* :angle-vector-sequence-raw avs :fast
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
      (send *ri* :angle-vector-sequence-raw avs :fast
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
    (let ((cube (gethash bin bin-cubes-))
          (base-name (send (send *baxter* :base_lk) :name)))
      (send *co* :add-object cube :frame-id base-name
            :relative-pose (send cube :copy-worldcoords)
            :object-id (format nil "bin_~A" (symbol-string bin)))))
  (:delete-bin-scene (bin)
    (send *co* :delete-object (gethash bin bin-cubes-)))
  (:add-shelf-scene ()
    (dolist (bin (list :a :b :c))
      (send self :add-bin-scene bin)))
  (:delete-shelf-scene ()
    (dolist (bin (list :a :b :c))
      (send self :delete-bin-scene bin)))
  (:add-cardboard-scene ()
    (let (cardboard-cube (base-name (send (send *baxter* :base_lk) :name)))
      (dolist (key (list :upper :lower))
        (setq cardboard-cube (gethash key cardboard-cubes-))
        (send *co* :add-object cardboard-cube :frame-id base-name
              :relative-pose (send cardboard-cube :copy-worldcoords)
              :object-id (format nil "~a_cardboard" (symbol2str key))))))
  (:delete-cardboard-scene ()
    (let (cardboard-cube)
      (dolist (key (list :upper :lower))
        (setq cardboard-cube (gethash key cardboard-cubes-))
        (send *co* :delete-object cardboard-cube))))
  (:add-tote-scene (arm)
    (let ((cube (gethash arm tote-cubes-))
          (base-name (send (send *baxter* :base_lk) :name)))
      (send *co* :add-object cube :frame-id base-name
            :relative-pose (send cube :copy-worldcoords)
            :object-id (format nil "~A_tote" (arm2str arm)))))
  (:delete-tote-scene (arm)
    (send *co* :delete-object (gethash arm tote-cubes-)))
  (:wipe-all-scene ()
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
      state))
  (:update-json (object &key src dst)
    (let ((service-name (format nil "/json_saver/update_json"))
          (req (instance jsk_arc2017_common::UpdateJSONRequest :init)))
      (ros::wait-for-service service-name)
      (if (consp src)
        (setq src (format nil "~a_~a" (symbol2str (car src)) (symbol-string (cdr src))))
        (setq src (symbol2str src)))
      (if (consp dst)
        (setq dst (format nil "~a_~a" (symbol2str (car dst)) (symbol-string (cdr dst))))
        (setq dst (symbol2str dst)))
      (send req :src src)
      (send req :dst dst)
      (send req :item object)
      (send (ros::service-call service-name req) :updated)))
  (:save-json ()
    (let ((service-name (format nil "/json_saver/save_json"))
          (req (instance std_srvs::TriggerRequest :init)))
      (ros::wait-for-service service-name)
      (send (ros::service-call service-name req) :success))))

(defun jsk_arc2017_baxter::arc-init (&key (ctype :default-controller) (moveit nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance jsk_arc2017_baxter::baxter-moveit-environment))
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
