#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter-interface.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")

(ros::load-ros-manifest "jsk_2016_01_baxter_apc")


;; ------------------------------------------------------------------------------------------------
;; utils
;; ------------------------------------------------------------------------------------------------

(defun polygon->center (polygon-msg)
  (send
    (instance pointcloud :init :points
              (mapcar #'(lambda (p)
                          (float-vector (send p :x) (send p :y) (send p :z)))
                      (send polygon-msg :polygon :points))) :centroid))


;; ------------------------------------------------------------------------------------------------
;; collector class
;; ------------------------------------------------------------------------------------------------

(defclass in-hand-data-collection
  :super propertied-object
  :slots
  (
   arm_
   object-id_
   object-list_
   save_service_client_
   grasping-log-filename_
   )
  )

(defmethod in-hand-data-collection
  (:init
    ()
    (if (string= (ros::get-param "~arm") "left")
      (setq arm_ :larm)
      (setq arm_ :rarm))
    (setq object-list_ (ros::get-param "~object_list"))
    (if (eq arm_ :larm)
      (jsk_2016_01_baxter_apc::baxter-init :ctype :larm-controller)
      (jsk_2016_01_baxter_apc::baxter-init :ctype :rarm-controller))
    ;; Initialize grasping log file
    (setq grasping-log-filename_ (pathname (ros::get-param "~grasping_log_filename")))
    (with-open-file
      (f grasping-log-filename_ :direction :output :if-exists :append :if-does-not-exist :create)
      (format f "timestamp,object_name,grasp_success~%"))
    (setq *robot* *baxter*)
    ;; Set slots
    (setq object-id_ 0)
    (if (eq arm_ :larm)
      (setq bin-list_ '(:a :d :g :j))
      (setq bin-list_ '(:c :f :i :l)))
    (setq object-pos_ (float-vector 750 -798 -180))
    ;; Wait for other nodes
    (ros::wait-for-service (format nil "~a_image_publisher/set_parameters" (arm-to-str arm_)))
    )
  (:wait-for-user-input
    ()
    (ros::ros-info "[~a] Waiting for user input to start" (ros::get-name))
    (let (yes)
      (while
        (not yes)
        (setq yes (send (ros::service-call "rqt_yn_btn" (instance jsk_rqt_plugins::YesNoRequest)) :yes))
        )
      )
    )
  (:send-save-request
    ()
    (ros::ros-info "[~a] Sending save request to data saving server" (ros::get-name))
    (let ((req (instance std_srvs::TriggerRequest :init)))
      (while
        (not
          (send
            (ros::service-call (format nil "data_collection_server_~a/save_request"
                                       (arm-to-str arm_)) req)
            :success))
        (unix::sleep 1)
        )
      )
    )
  (:view-hand-pose
    (&key (pos :random))
    (if (eq pos :random)
      (send *baxter* :view-hand-pose arm_ (elt bin-list_ (random (length bin-list_))))
      (send *baxter* :view-hand-pose arm_ pos)
      )
    (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    (unix::sleep 1)
    )
  (:move-arm-to-view-table
    ()
    (if (eq arm_ :larm)
      (send *baxter* :larm :angle-vector
            #f(9.25049 42.3633 -132.869 125.244 140.471 107.71 -19.1602 0.0))
      (send *baxter* :rarm :angle-vector
            #f(-9.25049 42.3633 132.869 125.244 -140.471 107.71 19.1602 0.0))
      )
    (let ((random-x 20)
          (random-y 20)
          (random-z 20))
      (send *baxter* arm_ :move-end-pos
            (float-vector (random random-x) (random random-y) (random random-z))
            :local
            :use-gripper t
            :revert-if-fail nil)
      )
    (send *ri* :gripper-servo-on arm_)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    )
  (:detect-object-on-table
    ()
    (send self :move-arm-to-view-table)
    (let* ((msg) (frame_id) (pose))
      (setq msg (one-shot-subscribe
                  (format nil
                          "cpi_decomposer_~a_table/centroid_pose_array"
                          (arm-to-str arm_))
                  geometry_msgs::PoseArray
                  :after-stamp (ros::time-now)))
      (if (or (null msg) (null (send msg :poses)))
        (progn
          (ros::ros-warn "[~a] No object is found on table. Retrying.." (ros::get-name))
          (send self :detect-object-on-table)
          )
        (progn
          (setq frame_id (send msg :header :frame_id))
          (setq pose (elt (send msg :poses) 0))
          (make-coords
            :pos (send (send *ri* :tf-pose->coords frame_id pose) :worldpos))))
      ))
  (:pick-object-from-table
    ()
    (ros::ros-info "[~a] Picking object from table" (ros::get-name))
    (let ((av0) (av1) (centroid (send self :detect-object-on-table)))
      (send *ri* :gripper-servo-on arm_)
      (setq av0 (send *baxter* arm_
                     :inverse-kinematics
                     (send (send centroid :copy-worldcoords)
                           :translate (float-vector 0 0 230) :world)
                     :use-gripper t
                     :rotation-axis nil))
      (unless av0 (return-from :pick-object-from-table))
      (send *ri* :angle-vector av0 3000)
      (send *ri* :wait-interpolation)
      (setq av1
            (send *baxter* arm_ :inverse-kinematics
                  (send (send centroid :copy-worldcoords)
                        :translate (float-vector 0 0 -50) :world)
                  :use-gripper t
                  :rotation-axis :z))
      (unless av1 (return-from :pick-object-from-table))
      (send *ri* :angle-vector av1 3000)
      (send *ri* :wait-interpolation)
      (send *ri* :start-grasp arm_)
      (unix::sleep 2)
      (send *ri* :gripper-servo-off arm_)
      (unix::sleep 1)
      (send *ri* :angle-vector av0 3000)
      (send *ri* :wait-interpolation)
      )
    (if (send *ri* :graspingp arm_)
      (progn
        (send self :write-grasping-log :success t)
        (send *ri* :angle-vector
              (send *baxter* :fold-to-keep-object arm_)
              3000)
        (send *ri* :wait-interpolation)
        (send *ri* :gripper-servo-on arm_)
        t)
      (progn
        (send self :write-grasping-log :success nil)
        (send *ri* :stop-grasp arm_)
        (unix::sleep 5)
        nil)
      )
    )
  (:write-grasping-log
    (&key (success))
    ;; ex)
    ;; timestamp,object_name,grasp_success
    ;; 14XXXX.XXXXX,oreo_mega_stuf,1
    ;; 14XXXX.XXXXX,oreo_mega_stuf,0
    (with-open-file (f grasping-log-filename_ :direction :output :if-exists :append)
      (format f "~a,~a,~a~%"
              (send (ros::time-now) :to-nsec)
              (elt object-list_ object-id_)
              (if success 1 0))
      )
    )
  (:detect-table-center
    ()
    (send self :move-arm-to-view-table)
    (let* ((msg)
           (center)
           (pose (instance geometry_msgs::Pose :init))
           (plane-center))
      (setq msg (one-shot-subscribe
                  (format nil
                          "multi_plane_estimate_~a_hand_camera/output_refined_polygon"
                          (arm-to-str arm_))
                  jsk_recognition_msgs::PolygonArray))
      (if (or (null msg) (null (send msg :polygons)))
        (progn
          (ros::ros-warn "[~a] No plane is found. Retrying.." (ros::get-name))
          (send self :detect-table-center))
        (progn
          (setq center (polygon->center (elt (send msg :polygons) 0)))
          (send pose :position :x (elt center 0))
          (send pose :position :y (elt center 1))
          (send pose :position :z (elt center 2))
          (send pose :orientation :w 1)
          (setq plane-center (send *ri* :tf-pose->coords (send msg :header :frame_id) pose))
          (make-coords :pos (send plane-center :worldpos))))
      ))
  (:place-object-on-table
    ()
    (ros::ros-info "[~a] Placing object on table" (ros::get-name))
    (send *ri* :gripper-servo-on arm_)
    (let (av0)
      (setq av0
            (send *baxter* arm_
                  :inverse-kinematics
                  (send (send (send self :detect-table-center) :copy-worldcoords)
                        :translate (float-vector 0 0 230) :world)
                  :use-gripper t)
            )
      (unless av0 (return-from :place-object-on-table))
      (send *ri* :angle-vector av0 3000)
      (send *ri* :wait-interpolation)
      )
    (send *ri* :stop-grasp arm_)
    (unix::sleep 5)
    )
  (:collect-data-for-an-object
    ()
    (ros::ros-info "[~a] Collecting data for an object" (ros::get-name))
    (dotimes (_ 4)
      ;; Pick object from table
      (while
        (null (send self :pick-object-from-table))
        (ros::ros-warn "[~a] Failed to pick object from table. Retrying.." (ros::get-name))
        (unix::sleep 3)
         )
      (unix::system (format nil "rosrun dynamic_reconfigure dynparam set_from_parameters data_collection_server_~a _save_dir:='~/.ros/jsk_2016_01_baxter_apc/in_hand_data_collection/picked_~a" (arm-to-str arm_) (send (ros::time-now) :to-nsec)))
      (unix::sleep 2)
      ;; View hand in a static position at first
      (if (eq arm_ :larm)
        (send self :view-hand-pose :pos :g)
        (send self :view-hand-pose :pos :i)
        )
      (send self :set-param-hand-pose)
      (if (send *ri* :graspingp arm_)
        (ros::set-param "~object_label" (elt object-list_ object-id_))
        (ros::set-param "~object_label" "no_object")
        )
      (send self :send-save-request)
      ;; View hand randomly (optimally in future)
      ;; TODO(wkentaro): view hand optimally here with ML
      (dotimes (_ 4)
        ;; View hand pose
        (send self :view-hand-pose)
        (send self :set-param-hand-pose)
        ;; Send saving request
        (if (send *ri* :graspingp arm_)
          (ros::set-param "~object_label" (elt object-list_ object-id_))
          (ros::set-param "~object_label" "no_object")
          )
        (send self :send-save-request)
        )
      (while
        (null (send self :place-object-on-table))
        (ros::ros-warn "[~a] Failed to place object on table. Retrying.." (ros::get-name))
        (unix::sleep 1)
        )
      )
    )
  (:request-next-object
    ()
    (let ((object-name (elt object-list_ object-id_)))
      (ros::ros-info "[~a] Requesting next object: '~a'" (ros::get-name) object-name)
      (ros::set-param "~object_label" object-name)
      (ros::set-dynparam
        (format nil "~a_image_publisher" (arm-to-str arm_))
        (list
          (cons "file_name"
                (format nil (ros::resolve-ros-path "package://jsk_apc2016_common/models/~a/image.jpg") object-name))))
      (unix::sleep 5)
      (send self :wait-for-user-input)
      )
    )
  (:set-param-hand-pose
    ()
    (let* ((camera-frame "kinect2_torso_rgb_optical_frame")
           (end-coords
             (send
               (send (*ri* . _tfl) :lookup-transform camera-frame "base" (ros::time 0))
               :transform (send *baxter* arm_ :end-coords :worldcoords)))
           (pos (ros::pos->tf-point (send end-coords :pos)))
           (quaternion (ros::rot->tf-quaternion (send end-coords :rot))))
      (ros::set-param "~hand_pose/frame_id" camera-frame)
      (ros::set-param "~hand_pose/position/x" (send pos :x))
      (ros::set-param "~hand_pose/position/y" (send pos :y))
      (ros::set-param "~hand_pose/position/z" (send pos :z))
      (ros::set-param "~hand_pose/orientation/x" (send quaternion :x))
      (ros::set-param "~hand_pose/orientation/y" (send quaternion :y))
      (ros::set-param "~hand_pose/orientation/z" (send quaternion :z))
      (ros::set-param "~hand_pose/orientation/w" (send quaternion :w))
      )
    )
  (:mainloop
    ()
    ;; reset-pose
    (send *ri* :angle-vector
          (send *baxter* :reset-pose arm_)
          5000)
    (send *ri* :wait-interpolation)
    (send *ri* :calib-pressure-threshold)
    (while
      (ros::ok)
      ;; Show next object
      (send self :request-next-object)
      ;; Collect
      (send self :collect-data-for-an-object)
      ;; Prepare for next object
      (send *ri* :angle-vector
            (send *baxter* :reset-pose arm_)
            5000)
      (send *ri* :wait-interpolation)
      (incf object-id_)
      )
    )
  )

;; __main__
(ros::roseus "in_hand_data_collection_main" :anonymous nil)
(setq collection (instance in-hand-data-collection :init))
(send collection :mainloop)
