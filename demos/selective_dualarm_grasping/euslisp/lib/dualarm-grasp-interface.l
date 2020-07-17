#!/usr/bin/env roseus

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/stow-interface.l")

(ros::load-ros-manifest "dualarm_grasping")

(unless (find-package "DUALARM_GRASPING")
  (make-package "DUALARM_GRASPING"))

(defclass dualarm_grasping::dualarm-grasp-interface
  :super jsk_arc2017_baxter::stow-interface
  :slots (grasping-way-
          cls-result-
          fail-recognize-arm-
          is-target-
          recognize-arm-
          tote-cube-
          target-label-
          target-tote-cube-
          use-scale-))

(defmethod dualarm_grasping::dualarm-grasp-interface
  (:init (&key (moveit t) (scale nil))
    (send-super :init :moveit moveit)
    (setq grasp-style- :suction)
    (setq grasping-way- :dual)
    (setq use-scale- scale))
  (:recognize-bboxes ()
    (ros::ros-info "[main] recognizing target tote")
    (send self :recognize-target-tote-box :stamp (ros::time-now))
    (ros::ros-info "[main] recognizing tote")
    (send self :recognize-tote-box :stamp (ros::time-now)))
  (:recognize-tote-box
    (&key (stamp (ros::time-now)))
    (let ((bbox-topic "/transformable_tote_markers/output/boxes")
                      bbox-msg bbox)
      (setq bbox-msg (one-shot-subscribe bbox-topic
                                         jsk_recognition_msgs::BoundingBoxArray
                                         :timeout 10000
                                         :after-stamp stamp))
      (if bbox-msg
        (progn
          (ros::ros-info "[~a] [:recognize-tote-box] recognize tote bbox" (ros::get-name))
          (setq bbox (car (send bbox-msg :boxes)))
          (setq tote-cube- (send self :bbox->cube bbox)))
        (ros::ros-fatal "[:recognize-tote-box] cannot recognize tote bbox"))))
  (:recognize-target-tote-box
    (&key (stamp (ros::time-now)))
    (let ((bbox-topic "/transformable_target_tote_markers/output/boxes")
                      bbox-msg bbox)
      (setq bbox-msg (one-shot-subscribe bbox-topic
                                         jsk_recognition_msgs::BoundingBoxArray
                                         :timeout 10000
                                         :after-stamp stamp))
      (if bbox-msg
        (progn
          (ros::ros-info "[~a] [:recognize-target-tote-box] recognize target tote bbox" (ros::get-name))
          (setq bbox (car (send bbox-msg :boxes)))
          (setq target-tote-cube- (send self :bbox->cube bbox)))
        (ros::ros-fatal "[:recognize-target-tote-box] cannot recognize target tote bbox"))))
  (:add-tote-scene ()
    (let ((base-name (send (send *baxter* :base_lk) :name)))
      (send *co* :add-object tote-cube- :frame-id base-name
            :relative-pose (send tote-cube- :copy-worldcoords)
            :object-id "tote")))
  (:delete-tote-scene ()
    (send *co* :delete-object tote-cube-))
  (:add-target-tote-scene ()
    (let ((base-name (send (send *baxter* :base_lk) :name)))
      (send *co* :add-object target-tote-cube- :frame-id base-name
            :relative-pose (send target-tote-cube- :copy-worldcoords)
            :object-id "target_tote")))
  (:delete-target-tote-scene ()
    (send *co* :delete-object target-tote-cube-))
  (:wait-for-user-input ()
    (let (can-start)
      (when moveit-p-
        (send self :add-target-tote-scene)
        (send self :add-tote-scene))
      (send *ri* :angle-vector (send *baxter* :arc-reset-pose) 3000 nil 0)
      (ros::ros-info "[:wait-for-user-input] wait for user input to start")
      (ros::wait-for-service "/rviz/yes_no_button")
      (setq can-start
            (send (ros::service-call "/rviz/yes_no_button"
                                     (instance jsk_gui_msgs::YesNoRequest))
                  :yes))
      (ros::ros-info "[:wait-for-user-input] received user input")
      (setq start-time- (ros::time-now))
      can-start))
  (:recognize-object (arm &key (trial-times 5))
    (let (is-recognized recognition-count)
      (setq trial-fail-count- 0)
      (setq label-names
            (ros::get-param (format nil "/~a_hand_camera/dualarm_grasp_segmentation/label_names"
                                    (arm2str arm))))
      (send *ri* :move-hand arm
            (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000 :wait nil)
      (ros::ros-info "[main] Recognizing objects in tote")
      (unless (> start-picking-fail-count- 0)
        (send self :move-arm-body->tote-overlook-pose arm)
        (send *ri* :wait-interpolation))
      (setq recognition-count 1)
      (let ((stamp (ros::time-now)))
        (while (null (or (> recognition-count trial-times) is-recognized))
          (setq is-recognized
            (send self :recognize-target-object arm :stamp stamp
                  :timeout (* recognition-count 10)))
          (when (and is-recognized
                     (eq grasping-way- :dual)
                     (< (length (gethash arm object-boxes-)) 2))
            (setq is-recognized nil))
          (incf recognition-count)))
      (when is-recognized
        (setq target-label- (elt (send cls-result- :labels) 0))
        (setq target-obj- (elt label-names target-label-)))
      is-recognized))
  (:move-arm-body->tote-overlook-pose
    (arm &key (gripper-angle 90))
    (let (avs offset rpy
              (offset-x (if (eq arm :larm) 0 -0))
              (offset-y (if (eq arm :larm) 250 -250)))
      (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
      (send *baxter* :reset-pose arm)
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (setq offset (float-vector offset-x offset-y 250))
      (setq rpy (float-vector 0 pi/2 (if (eq arm :larm) pi/2 -pi/2)))
      (pushback
        (send self :ik->tote-center arm
              :offset offset :rpy rpy :use-gripper nil)
        avs)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000
            (send *ri* :get-arm-controller arm) 0)))
  (:ik->tote-center
    (arm &key (offset #f(0 0 0)) (rpy #f(0 0 0))
         (rotation-axis t) (use-gripper nil) (move-palm-end nil))
    (let (tote-coords)
      (setq tote-coords (send tote-cube- :copy-worldcoords))
      (send tote-coords :translate
            (float-vector 0.0 0.0 (/ (z-of-cube tote-cube-) 2.0))
            :local)
      (send tote-coords :translate offset :world)
      (send tote-coords :rotate (aref rpy 0) :z)
      (send tote-coords :rotate (aref rpy 1) :y)
      (send tote-coords :rotate (aref rpy 2) :x)
      (send *baxter* arm :inverse-kinematics tote-coords
            :rotation-axis rotation-axis
            :use-gripper use-gripper
            :move-palm-end move-palm-end)))
  (:recognize-target-object
    (arm &key (stamp (ros::time-now)) (timeout 10))
    (let ((box-topic
            (format nil "~a_hand_camera/cluster_indices_decomposer/boxes" (arm2str arm)))
          (com-topic
            (format nil "~a_hand_camera/cluster_indices_decomposer/centroid_pose_array" (arm2str arm)))
          (cls-topic
            (format nil "~a_hand_camera/dualarm_grasp_segmentation/output/grasp_class" (arm2str arm)))
          box-msg com-msg cls-msg obj-box obj-coords cls-style is-recognized)
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
      (ros::subscribe cls-topic dualarm_grasping::GraspClassificationResult
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq cls-msg msg)))))
      (while (and (not (and com-msg cls-msg box-msg
                         (if (equal (send cls-msg :style) "single")
                           (and (> (length (send box-msg :boxes)) 0)
                                (> (length (send com-msg :poses)) 0))
                           (and (> (length (send box-msg :boxes)) 1)
                                (> (length (send com-msg :poses)) 1)))))
                  (> (+ (send stamp :to-sec) timeout) (send (ros::time-now) :to-sec)))
             (unix::usleep (* 50 1000))
             (ros::spin-once))
      (ros::unsubscribe box-topic)
      (ros::unsubscribe com-topic)
      (ros::unsubscribe cls-topic)
      (cond
        ((and box-msg com-msg cls-msg)
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
             (setq is-recognized nil)))
         (setq cls-style (send cls-msg :style))
         (setq grasping-way-
               (cond ((equal cls-style "dual") :dual)
                     ((equal cls-style "single") :single)
                     (t nil)))
         (setq is-target- (send cls-msg :is_target))
         (setq cls-result- (send cls-msg :classification)))
        (t
         (ros::ros-error "[:recognize-target-object] arm: ~a failed to get cpi msg" arm)
         (ros::ros-error "[:recognize-target-object] arm: ~a box-msg: ~a, com-msg ~a, cls-msg ~a"
                         arm (if box-msg t nil) (if com-msg t nil) (if cls-msg t nil))
         (setq is-recognized nil)))
      is-recognized))
  (:set-fail-recognize-arm (arm)
    (setq fail-recognize-arm- arm))
  (:set-recognize-arm (arm &key (update nil))
    (if (and update fail-recognize-arm-)
      (setq recognize-arm- (if (eq fail-recognize-arm- :larm) :rarm :larm))
      (setq recognize-arm- arm))
    recognize-arm-)
  (:get-is-target () is-target-)
  (:get-recognize-arm () recognize-arm-)
  (:get-grasp-arm (arm)
    (cond ((eq grasping-way- :dual) :arms)
          ((eq grasping-way- :single) arm)
          (t nil)))
  (:pick-object (arm)
    (send *ri* :speak
          (format nil "robot is picking ~a." (underscore-to-space target-obj-)))
    (ros::ros-info-green "grasping-way: ~A, object: ~A" grasping-way- target-obj-)
    (let (pick-result graspingp)
      (setq recognize-fail-count- 0)
      (when moveit-p- (send self :delete-tote-scene))
      (send self :set-movable-region-for-tote :larm :offset (list 50 50 0))
      (send self :set-movable-region-for-tote :rarm :offset (list 50 50 0))
      (when use-scale- (send self :reset-scale arm))
      (send *baxter* :head_pan :joint-angle
            (cond ((eq arm :larm) -70)
                  ((eq arm :rarm) 70)
                  ((eq arm :arms ) 0)))
      (setq grasp-style- :suction)
      (setq pick-result
            (send self :pick-object-in-tote arm
                  :n-trial 2
                  :n-trial-same-pos 1
                  :do-stop-grasp nil
                  :grasp-style grasp-style-))
      (when moveit-p- (send self :add-tote-scene))
      ;; Don't trust pressure sensor
      ;; (unless (eq pick-result :ik-failed)
      ;;   (setq graspingp (send *ri* :graspingp arm grasp-style-))
      ;;   (unless graspingp (return-from :pick-object nil))
      ;;   )
      (if (eq arm :arms)
        (progn
          (send *baxter* :larm :move-end-pos #f(0 0 100) :world :rotation-axis t)
          (send *baxter* :rarm :move-end-pos #f(0 0 100) :world :rotation-axis t))
        (send *baxter* arm :move-end-pos #f(0 0 100) :world :rotation-axis t))
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
      (send *ri* :wait-interpolation)
      (unless (eq pick-result :grasp-succeeded)
        (send *ri* :stop-grasp arm)
        (return-from :pick-object pick-result))
      ;; Don't trust pressure sensor
      ;; (setq graspingp (send *ri* :graspingp arm grasp-style-))
      ;; (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
      ;; graspingp))
      (if use-scale-
        (progn
          (setq scale-candidates-
                (send self :get-scale-candidates arm :picked :stamp (ros::time-now)))
          ;; If scale value isn't changed, we don't have to verify-object and return-object
          (if (eq scale-candidates- :no-change)
            (progn
              (ros::ros-info-green "[main] arm ~a: grasping nothing is detected by scale" arm)
              nil) pick-result))
        pick-result)))
  (:set-movable-region-for-tote
    (arm &key (offset (list 0 0 0)))
    (sethash arm tote-movable-regions-
             (send self :cube->movable-region tote-cube- :offset offset)))
  (:pick-object-in-tote (arm &rest args)
    (let (pick-result)
      (setq pick-result
        (if (eq arm :arms)
          (send* self :dualarm-pick-object-in-tote args)
          (send-super* :pick-object-in-tote arm args)))
    pick-result))
  (:dualarm-pick-object-in-tote
    (&key (n-trial 1) (n-trial-same-pos 1) (do-stop-grasp nil) (grasp-style :suction))
    (let (pick-result movable-region
          larm-obj-pos larm-obj-cube larm-object-index
          rarm-obj-pos rarm-obj-cube rarm-object-index)
      (setq larm-movable-region (gethash :larm tote-movable-regions-))
      (setq rarm-movable-region (gethash :rarm tote-movable-regions-))
      (unless (and rarm-movable-region larm-movable-region)
        (ros::ros-error "[:dualarm-pick-object-in-tote] No movable region for tote. Call :set-movable-region-for-tote first.")
        (return-from :dualarm-pick-object-in-tote nil))
      (unless (> (length (gethash recognize-arm- object-coords-)) 1)
        (return-from :dualarm-pick-object-in-tote nil))
      (unless (> (length (gethash recognize-arm- object-boxes-)) 1)
        (return-from :dualarm-pick-object-in-tote nil))
      (if (> (elt (send (elt (gethash recognize-arm- object-coords-) 0) :pos) 1)
             (elt (send (elt (gethash recognize-arm- object-coords-) 1) :pos) 1))
        (setq larm-object-index 0
              rarm-object-index 1)
        (setq larm-object-index 1
              rarm-object-index 0))
      (setq larm-obj-pos
            (send self :get-object-position recognize-arm-
                  larm-movable-region :object-index larm-object-index))
      (setq larm-obj-cube
            (send self :bbox->cube
                  (elt (gethash recognize-arm- object-boxes-) larm-object-index)))
      (setq rarm-obj-pos
            (send self :get-object-position recognize-arm-
                  rarm-movable-region :object-index rarm-object-index))
      (setq rarm-obj-cube
            (send self :bbox->cube
                  (elt (gethash recognize-arm- object-boxes-) rarm-object-index)))
      (send *ri* :stop-grasp :larm :pinch)
      (send *ri* :stop-grasp :rarm :pinch)
      (send *ri* :calib-proximity-threshold :larm)
      (send *ri* :calib-proximity-threshold :rarm)
      (send *ri* :gripper-servo-on :larm)
      (send *ri* :gripper-servo-on :rarm)
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (when (or (null pick-result) (null (eq pick-result :grasp-succeeded)))
            (setq pick-result
                  (send self :dualarm-try-to-pick-object larm-obj-pos rarm-obj-pos 0 0
                        :offset (float-vector 0 0 (- (* i -30) 30)))))))
      (when do-stop-grasp
        (unless (eq pick-result :grasp-succeeded)
          (send *ri* :stop-grasp :larm)
          (send *ri* :stop-grasp :rarm)))
      ; (send *ri* :move-hand :larm
      ;       (send *baxter* :hand-grasp-pre-pose :larm :opposed) 1000 :wait nil)
      ; (send *ri* :move-hand :rarm
      ;       (send *baxter* :hand-grasp-pre-pose :rarm :opposed) 1000 :wait nil)
      ; (send *ri* :wait-interpolation)
      (send *ri* :gripper-servo-on :larm)
      (send *ri* :gripper-servo-on :rarm)
      pick-result))
  (:dualarm-try-to-pick-object
    (larm-obj-pos rarm-obj-pos larm-suction-yaw rarm-suction-yaw &key (offset #f(0 0 0)))
    (let (larm-graspingp rarm-grasingp av
          (larm-coords-before-approach (send *baxter* :larm :end-coords :copy-worldcoords))
          (rarm-coords-before-approach (send *baxter* :rarm :end-coords :copy-worldcoords))
          larm-prismatic-angle rarm-prismatic-angle
          larm-ik-success rarm-ik-success moveit-success previous-av)
      (send *ri* :move-hand :larm
            (send *baxter* :hand-grasp-pre-pose :larm :cylindrical) 1000 :wait nil)
      (send *ri* :move-hand :rarm
            (send *baxter* :hand-grasp-pre-pose :rarm :cylindrical) 1000 :wait t)
      ; (send *ri* :move-hand :larm
      ;       (send *baxter* :hand-grasp-pose :larm :spherical :angle 90) 1000 :wait nil)
      ; (send *ri* :move-hand :rarm
      ;       (send *baxter* :hand-grasp-pose :rarm :spherical :angle 90) 1000 :wait nil)
      ; (send *ri* :wait-interpolation)
      ; (send *ri* :move-hand :larm (send *baxter* :hand-grasp-pose :larm :cylindrical) 1000)
      ; (send *ri* :move-hand :rarm (send *baxter* :hand-grasp-pose :rarm :cylindrical) 1000)
      ; (send *ri* :wait-interpolation)
      (send larm-coords-before-approach :locate (v+ larm-obj-pos #f(0 0 300)) :world)
      (send rarm-coords-before-approach :locate (v+ rarm-obj-pos #f(0 0 300)) :world)
      (ros::ros-info "[:dualarm-try-to-pick-object] larm-suction-yaw: ~a rarm-suction-yaw: ~a" larm-suction-yaw rarm-suction-yaw)
      ;; start the vacuum gripper before approaching to the object
      (setq previous-av (send *baxter* :angle-vector))
      (setq larm-ik-success
            (send *baxter* :larm :inverse-kinematics
                  (make-coords :pos (v+ larm-obj-pos #f(0 0 150))
                               :rpy (float-vector larm-suction-yaw 0 0))
                  :use-gripper t
                  :rotation-axis t))
      (setq rarm-ik-success
            (send *baxter* :rarm :inverse-kinematics
                  (make-coords :pos (v+ rarm-obj-pos #f(0 0 150))
                               :rpy (float-vector rarm-suction-yaw 0 0))
                  :use-gripper t
                  :rotation-axis t))
      (unless (and larm-ik-success rarm-ik-success)
        (ros::ros-error "[:dualarm-try-to-pick-object] IK failed")
        (send *baxter* :angle-vector previous-av)
        (return-from :dualarm-try-to-pick-object :ik-failed))
      (setq moveit-success
            (send *ri* :angle-vector (send *baxter* :angle-vector) 3000 nil 0))
      (unless moveit-success
        (ros::ros-error "[:dualarm-try-to-pick-object] Moveit failed")
        (send *baxter* :angle-vector previous-av)
        (return-from :dualarm-try-to-pick-object :moveit-failed))
      (send *ri* :wait-interpolation)

      (ros::ros-info "[:dualarm-try-to-pick-object] start vacuum gripper")
      (send *ri* :start-grasp :larm)
      (send *ri* :start-grasp :rarm)
      ;; suction: prismatic-based approach
      (send *baxter* :slide-gripper :larm 120 :relative nil)
      (send *baxter* :slide-gripper :rarm 120 :relative nil)
      (setq larm-ik-success
            (send *baxter* :larm :inverse-kinematics
                  (make-coords :pos (v+ larm-obj-pos offset)
                               :rpy (float-vector larm-suction-yaw 0 0))
                  :use-gripper t
                  :rotation-axis t))
      (setq rarm-ik-success
            (send *baxter* :rarm :inverse-kinematics
                  (make-coords :pos (v+ rarm-obj-pos offset)
                               :rpy (float-vector rarm-suction-yaw 0 0))
                  :use-gripper t
                  :rotation-axis t))
      (unless (and larm-ik-success rarm-ik-success)
        (ros::ros-error "[:dualarm-try-to-pick-object] IK failed")
        (send *baxter* :angle-vector previous-av)
        (return-from :dualarm-try-to-pick-object :ik-failed))
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
      (send *ri* :wait-interpolation)
      ; (unless moveit-success
      ;   (ros::ros-error "[:dualarm-try-to-pick-object] Moveit failed")
      ;   (send *baxter* :angle-vector previous-av)
      ;   (return-from :dualarm-try-to-pick-object :moveit-failed))

      (setq larm-prismatic-angle (send *baxter* :larm :gripper-x :joint-angle))
      (setq rarm-prismatic-angle (send *baxter* :rarm :gripper-x :joint-angle))
      (send *baxter* :slide-gripper :larm 0 :relative nil)
      (send *baxter* :slide-gripper :rarm 0 :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
      (send *ri* :wait-interpolation)  ;; move down only the hand palm

      (send *baxter* :slide-gripper :larm larm-prismatic-angle :relative nil)
      (send *baxter* :slide-gripper :rarm rarm-prismatic-angle :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
      ; FIXME: :wait-interpolation-until using :prismatic-loaded sometimes ends too fast,
      ; so currently we only check :grasp (suction pressure).
      (send self :wait-interpolation-until-grasp-dualarm :grasp)

      (send *baxter* :slide-gripper :larm 120 :relative nil)  ;; maximum angle of prismatic joint
      (send *baxter* :slide-gripper :rarm 120 :relative nil)  ;; maximum angle of prismatic joint
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
      (send self :wait-interpolation-until-grasp-dualarm :prismatic :grasp)

      (setq larm-graspingp (send *ri* :graspingp :larm :suction))
      (setq rarm-graspingp (send *ri* :graspingp :rarm :suction))
      (ros::ros-info "[:dualarm-pick-object-in-tote] arm:~a graspingp: ~a" :larm larm-graspingp)
      (ros::ros-info "[:dualarm-pick-object-in-tote] arm:~a graspingp: ~a" :rarm rarm-graspingp)

      (unless (and larm-graspingp rarm-grasingp)
        (ros::ros-info "[:dualarm-pick-object-in-tote] again approach to the object")
        (let ((temp-av (send *baxter* :angle-vector)))
          ;; only if robot can solve IK
          (if (and (send *baxter* :larm :move-end-pos #f(0 0 -150) :local)
                   (send *baxter* :rarm :move-end-pos #f(0 0 -150) :local))
            (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0))
          (send self :wait-interpolation-until-grasp-dualarm :prismatic :grasp)
          (send *ri* :angle-vector-raw (send *baxter* :angle-vector temp-av) 3000 nil 0)
          (send self :wait-interpolation-until-grasp-dualarm :prismatic :grasp)))
      ; ;; Open fingers in bin
      ; (send *ri* :move-hand :larm
      ;       (send *baxter* :hand-grasp-pre-pose :larm :spherical) 1000 :wait nil)
      ; (send *ri* :move-hand :rarm
      ;       (send *baxter* :hand-grasp-pre-pose :rarm :spherical) 1000 :wait nil)
      ; (send *ri* :wait-interpolation)

      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (send *baxter* :rotate-gripper :larm 30 :relative nil)
      (send *baxter* :rotate-gripper :rarm 30 :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000 nil 0)
      (send *ri* :wait-interpolation)

      ;; suction: prismatic-based approach
      (send *baxter* :slide-gripper :larm 50 :relative nil)
      (send *baxter* :slide-gripper :rarm 50 :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
      ;; suction: prismatic-based approach
      ;; lift object
      (ros::ros-info "[:dualarm-pick-object-in-tote] lift the object")
      (send *ri* :gripper-servo-off :larm)
      (send *ri* :gripper-servo-off :rarm)

      (let ((larm-tc (send *baxter* :larm :end-coords :copy-worldcoords))
            (rarm-tc (send *baxter* :rarm :end-coords :copy-worldcoords)))
        ;; overwrite only world-z
        (setf (aref (send larm-tc :worldpos) 2)
              (elt (send larm-coords-before-approach :worldpos) 2))
        (setf (aref (send rarm-tc :worldpos) 2)
              (elt (send rarm-coords-before-approach :worldpos) 2))
        (send *baxter* :larm :inverse-kinematics larm-tc :rotation-axis t)
        (send *baxter* :rarm :inverse-kinematics rarm-tc :rotation-axis t))
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
      (send *ri* :wait-interpolation)

      (setq graspingp (and (send *ri* :graspingp :larm :suction)
                           (send *ri* :graspingp :rarm :suction)))
      (ros::ros-info "[:dualarm-pick-object-in-tote] graspingp: ~a" graspingp)
      (if graspingp :grasp-succeeded :grasp-failed)))
  (:return-from-recognize-object (arm)
    (let ((arms (if (eq arm :arms) (list :larm :rarm) (list arm))))
      (dolist (tmp-arm arms)
        (ros::ros-info "[:return-from-recognize-object] ~A: return to arc-reset-pose" tmp-arm)
        (send *ri* :angle-vector (send *baxter* :arc-reset-pose tmp-arm) 5000 nil 0)
        (send *ri* :wait-interpolation))))
  (:return-from-pick-object (arm)
    (let ((arms (if (eq arm :arms) (list :larm :rarm) (list arm))))
      (dolist (tmp-arm arms)
        (ros::ros-info "[:return-from-pick-object] ~A: return to arc-reset-pose" tmp-arm)
        (send *ri* :angle-vector (send *baxter* :arc-reset-pose tmp-arm) 5000 nil 0)
        (send *ri* :wait-interpolation))))
  (:verify-object ()
    (let ((stamp (ros::time-now)))
      (if use-scale-
        (progn
          (when (eq scale-candidates- :timeout)
            (ros::ros-error "subscribed weight candidates, but timeout")
            (return-from :verify-object nil))
          (ros::ros-info-green "[main] arm ~a: scale candidates ~a" arm scale-candidates-)
          (unless scale-candidates- (return-from :verify-object nil))
          (find target-obj- scale-candidates- :test #'string=)
          (if target-obj- t nil))
        t)))
  (:place-object (arm &key (opposite nil) (distance 400))
    (send *ri* :speak
          (format nil "placing ~a in target tote." (underscore-to-space target-obj-)))
    (ros::ros-info "[main] place ~a in target tote" (underscore-to-space target-obj-))
    (when moveit-p- (send self :delete-target-tote-scene))
    (if (eq arm :arms)
      (progn
        (send *baxter* :larm :move-end-pos
              (float-vector (* (if opposite -1 1) distance) 0 0)
              :world :rotation-axis t :use-gripper t)
        (send *baxter* :rarm :move-end-pos
              (float-vector (* (if opposite -1 1) distance) 0 0)
              :world :rotation-axis t :use-gripper t))
      (progn
        (send *baxter* arm :inverse-kinematics
              (make-coords :pos (send (send *baxter* :larm :end-coords :copy-worldcoords) :worldpos)
                           :rpy (float-vector (if (eq arm :larm) -90 90) 0 0))
              :use-gripper t
              :rotation-axis t)
        (send *baxter* arm :move-end-pos
              (float-vector (* (if opposite -1 1) distance) 0 0)
              :world :rotation-axis :z :use-gripper t)))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
    (send *ri* :wait-interpolation)
    (if (eq arm :arms)
      (progn
        (send *baxter* :larm :move-end-pos #f(0 0 -150) :world :rotation-axis t :use-gripper t)
        (send *baxter* :rarm :move-end-pos #f(0 0 -150) :world :rotation-axis t :use-gripper t))
      (send *baxter* arm :move-end-pos #f(0 0 -150) :world :rotation-axis :z :use-gripper t))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp :arms)
    (if (eq arm :arms)
      (progn
        (send self :spin-off-by-wrist :larm :times 3)
        (send self :spin-off-by-wrist :rarm :times 3))
      (send self :spin-off-by-wrist arm :times 3))
    (send self :spin-off-by-wrist arm :times 3)
    (send *ri* :wait-interpolation)
    (ros::ros-info "[main] ~a, place object in bin" arm)
    (if (eq arm :arms)
      (progn
        (send *baxter* :larm :move-end-pos #f(0 0 150) :world :rotation-axis t :use-gripper t)
        (send *baxter* :rarm :move-end-pos #f(0 0 150) :world :rotation-axis t :use-gripper t))
      (send *baxter* arm :move-end-pos #f(0 0 150) :world :rotation-axis :z :use-gripper t))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
    (send *ri* :wait-interpolation)
    (when moveit-p- (send self :add-target-tote-scene))
    (send *ri* :angle-vector (send *baxter* :arc-reset-pose :arms) 3000 nil 0)
    (send *ri* :wait-interpolation))
  (:return-from-place-object (arm)
    (if (eq arm :arms)
      (progn
        (send-super :return-from-place-object :larm)
        (send-super :return-from-place-object :rarm))
      (send-super :return-from-place-object arm)))
  (:return-object (arm)
    (send *ri* :speak "returning object")
    (ros::ros-info "[main] return in tote")
    (when moveit-p- (send self :delete-tote-scene))
    (if (eq arm :arms)
      (progn
        (send *baxter* :larm :move-end-pos #f(0 0 -300) :world :rotation-axis t)
        (send *baxter* :rarm :move-end-pos #f(0 0 -300) :world :rotation-axis t))
      (send *baxter* arm :move-end-pos #f(0 0 -300) :world :rotation-axis :z))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp arm)
    (if (eq arm :arms)
      (progn
        (send self :spin-off-by-wrist :larm :times 3)
        (send self :spin-off-by-wrist :rarm :times 3))
      (send self :spin-off-by-wrist arm :times 3))
    (send *ri* :wait-interpolation)
    (ros::ros-info "[main] ~a, return object in tote" arm)
    (if (eq arm :arms)
      (progn
        (send *baxter* :larm :move-end-pos #f(0 0 300) :world :rotation-axis :z)
        (send *baxter* :rarm :move-end-pos #f(0 0 300) :world :rotation-axis :z))
      (send *baxter* arm :move-end-pos #f(0 0 300) :world :rotation-axis :z))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0)
    (send *ri* :wait-interpolation)
    (when moveit-p- (send self :add-tote-scene))
    (send *ri* :angle-vector (send *baxter* :arc-reset-pose :arms) 3000 nil 0)
    (send *ri* :wait-interpolation))
  (:calib-prismatic-joint (arm)
    (if (eq arm :arms)
      (progn
        (send-super :calib-prismatic-joint :larm)
        (send-super :calib-prismatic-joint :rarm))
      (send-super :calib-prismatic-joint arm)))
  (:giveup ()
     (let ((srv-name (format nil "/~a_hand_camera/dualarm_grasp_segmentation/get_another" (arm2str recognize-arm-)))
           (req (instance dualarm_grasping::GetAnotherRequest :init)))
      (ros::wait-for-service srv-name)
      (send req :style (if (eq grasping-way- :dual) "dual" "single"))
      (send req :label target-label-)
      (send (ros::service-call srv-name req) :success)))
  (:reset-giveup (arm)
     (let ((srv-name (format nil "/~a_hand_camera/dualarm_grasp_segmentation/reset" (arm2str arm)))
           (req (instance std_srvs::TriggerRequest :init)))
      (ros::wait-for-service srv-name)
      (send (ros::service-call srv-name req) :success)))
  (:set-grasping-way (grasping-way)
    (let ((dynsrv-name
            (format nil "/~a_hand_camera/dualarm_grasp_segmentation"
                    (arm2str recognize-arm-)))
          (dynparam-name "grasping_way"))
      (ros::set-dynparam dynsrv-name
                         (cons dynparam-name grasping-way))
      (setq grasping-way-
            (if (equal grasping-way "single") :single :dual))
      (ros::ros-info "set grasping-way: ~A" grasping-way)))
  (:recognition-save-request ()
    (let ((savesrv-name
            (format nil "/~a_hand_camera/before_grasping/data_collection/save_request"
                    (arm2str recognize-arm-))))
      (ros::wait-for-service savesrv-name)
      (if (send (ros::service-call savesrv-name (instance std_srvs::TriggerRequest :init)) :success)
        (ros::ros-info "[:recognition-save-request] success to save recognition data")
        (ros::ros-error "[:recognition-save-request] fail to save recognition data"))))
  (:result-save-request ()
    (let ((savesrv-name
            (format nil "/~a_hand_camera/after_grasping/data_collection/save_request"
                    (arm2str recognize-arm-))))
      (ros::wait-for-service savesrv-name)
      (if (send (ros::service-call savesrv-name (instance std_srvs::TriggerRequest :init)) :success)
        (ros::ros-info "[:result-save-request] success to save result data")
        (ros::ros-error "[:result-save-request] fail to save result data"))))
  (:set-save-dir ()
    (let* ((before-dynsrv-name
             (format nil "/~a_hand_camera/before_grasping/data_collection/"
                     (arm2str recognize-arm-)))
           (after-dynsrv-name
             (format nil "/~a_hand_camera/after_grasping/data_collection/"
                         (arm2str recognize-arm-)))
           (localtime (unix::localtime))
           (stamp (format nil "~A~0,2d~0,2d_~0,2d~0,2d~0,2d"
                           (+ (elt localtime 5) 1900)
                           (+ (elt localtime 4) 1)
                           (elt localtime 3) (elt localtime 2)
                           (elt localtime 1) (elt localtime 0)))
           (before-save-dir (format nil "~~/.ros/dualarm_grasping/recognition/~A" stamp))
           (after-save-dir (format nil "~~/.ros/dualarm_grasping/result/~A" stamp)))
      (ros::set-dynparam before-dynsrv-name (cons "save_dir" before-save-dir))
      (ros::set-dynparam after-dynsrv-name (cons "save_dir" after-save-dir))))
  (:set-result (pick-result)
    (ros::set-param "/result"
                    (cond ((eq pick-result :grasp-succeeded) "success")
                          ((eq pick-result :grasp-unstable) "unstable_grasp")
                          ((eq pick-result :grasp-failed) "drop")
                          ((eq pick-result :ik-failed) "ik_failed")
                          ((eq pick-result :moveit-failed) "moveit_failed")
                          (t "unknown"))))
  (:graspingp (arm)
    (if (eq arm :arms)
      (and (send *ri* :graspingp :larm :suction)
           (send *ri* :graspingp :rarm :suction))
      (send *ri* :graspingp arm :suction)))
  (:prismaticp (arm)
    (let ((prismatic-load (send *ri* :get-prismatic-load arm)))
      (and (< prismatic-load -0.07) (< (send *ri* :get-prismatic-vel arm) 0.01))))
  (:wait-interpolation-until-grasp-dualarm (&rest args)
    (let ((arms (list :larm :rarm))
          suctionp prismaticp)
      ;; wait for :interpolatingp
      (setq suctionp (memq :suction args))
      (setq prismaticp (memq :prismatic args))
      (dotimes (x 100)
        (if (send *ri* :interpolatingp) (return))
        (unix::usleep 1000))
      (while (send *ri* :interpolatingp)
        (dolist (tmp-arm arms)
          (when
            (cond
              ((null suctionp) (send self :prismaticp tmp-arm))
              ((null prismaticp) (send self :graspingp tmp-arm))
              (t (or (send self :graspingp tmp-arm) (send self :prismaticp tmp-arm))))
            (ros::ros-info "[:wait-interpolation-until-grasp-dualarm] Cancel angle vector: ~a" tmp-arm)
            (send *ri* :cancel-angle-vector
                  :controller-type (send *ri* :get-arm-controller tmp-arm))
            (setq arms (remove tmp-arm arms))))
        (send *ri* :spin-once)
        (unix::usleep 1000)))))


(defun dualarm_grasping::dualarm-grasp-init
  (&key (ctype :default-controller) (moveit nil) (scale nil) (collision-avoidance nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance jsk_arc2017_baxter::baxter-moveit-environment))
      (setq mvit-rb (instance jsk_arc2017_baxter::baxter-robot :init)))
    (unless (boundp '*ri*)
      (setq *ri* (instance jsk_arc2017_baxter::baxter-interface :init :type ctype
                           :moveit-environment mvit-env
                           :moveit-robot mvit-rb)))
    (unless (boundp '*baxter*)
      (if collision-avoidance
        (setq *baxter* (instance jsk_arc2017_baxter::baxter-robot-safe :init))
        (setq *baxter* (instance jsk_arc2017_baxter::baxter-robot :init))))
    (unless (boundp '*co*)
      (setq *co* (when moveit (instance collision-object-publisher :init))))
    (unless (boundp '*ti*)
      (setq *ti* (instance dualarm_grasping::dualarm-grasp-interface :init
                           :moveit moveit :scale scale)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)))
