#!/usr/bin/env roseus

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")
(require "package://grasp_prediction_arc2017/euslisp/hasegawa_iros2018/lib/pick-book-base-interface.l")


(ros::load-ros-package "jsk_gui_msgs")
(ros::load-ros-package "std_srvs")

(unless (find-package "GRASP_PREDICTION_ARC2017")
  (make-package "GRASP_PREDICTION_ARC2017"))

(defclass grasp_prediction_arc2017::pick-book-interface
  :super grasp_prediction_arc2017::pick-book-base-interface
  :slots (grasp-style-
          label-names-
          moveit-p-
          moving-object-p-
          picking-fail-count-
          place-target-cube-
          place-target-label-
          prev-target-obj-
          recognize-fail-count-
          start-picking-fail-count-
          target-bin-
          target-box-
          target-obj-
          tried-directions-))

(defmethod grasp_prediction_arc2017::pick-book-interface
  (:init (&key (moveit nil))
    (send-super :init)
    (setq picking-fail-count- 0)
    (setq tried-directions- nil)
    (setq recognize-fail-count- 0)
    (setq start-picking-fail-count- 0)
    (setq moveit-p- moveit)
    (setq grasp-style- nil))
  (:recognize-bboxes ()
    (ros::ros-info "[main] recognizing shelf bin boxes")
    (send self :recognize-bin-boxes :stamp (ros::time-now))
    (ros::ros-info "[main] recognizing cardboard boxes")
    (send self :recognize-cardboard-boxes :stamp (ros::time-now))
    ;; XXX: movable region will be set in :pick-object
    ;; (send self :set-movable-region-for-bin :offset (list 80 120 0))
    )
  (:wait-for-user-input (arm)
    (let (can-start)
      (when moveit-p-
        (send self :add-shelf-scene)
        ;; (send self :add-cardboard-rack-scene)
        (send self :add-cart-handle-scene)
        )
      (send *ri* :speak
            (format nil "~a arm is waiting for user input to start the picking task."
                    (arm2str arm)))
      (ros::ros-info "[:wait-for-user-input] wait for user input to start: ~a" arm)
      (ros::wait-for-service "/rviz/yes_no_button")
      (setq can-start
            (send (ros::service-call "/rviz/yes_no_button"
                                     (instance jsk_gui_msgs::YesNoRequest))
                  :yes))
      (ros::ros-info "[:wait-for-user-input] received user input: ~a" arm)
      (send self :get-work-orders arm)
      (send self :reset-object-tables :bins)
      (dolist (bin bin-list-)
        (send self :set-objects-param arm bin :object-type :all))
      can-start))
  (:calib-prismatic-joint (arm &key (timeout 10))
    (send *ri* :start-prismatic-calib arm)
    (let ((time-count 1))
      (while (and (send *ri* :prismatic-calibratingp arm)
                  (> (* timeout 10) time-count))
        (unix::usleep (* 100 1000))
        (incf time-count))
      (if (send *ri* :prismatic-calibratingp arm) nil t)))
  (:set-target (arm)
    (setq label-names-
          (ros::get-param (format nil "/~a_hand_camera/label_names"
                                  (arm2str arm))))
    ;; get target param
    (setq target-bin- (send self :get-next-target-bin arm))
    (unless (send self :check-bin-exist target-bin-)
      (ros::ros-warn "[~a] [main] could not find bin box: ~a" (ros::get-name) target-bin-)
      (return-from :set-target nil))
    (ros::set-dynparam
      (format nil "/~a_hand_camera/bbox_array_to_bbox" (arm2str arm))
      (cons "index" (position target-bin- bin-list-)))
    (send self :set-target-location arm (cons :bin target-bin-))
    (ros::set-param
      (format nil "~a_hand/target_bin" (arm2str arm))
      (symbol2str target-bin-))
    (unless target-bin-
      (ros::ros-info "there is no order to execute"))
    (if target-bin- t nil))
  (:recognize-object (arm &key (trial-times 10))
    (let (is-recognized recognition-count)
      (setq grasp-style- nil)
      (send *ri* :move-hand arm
            (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000 :wait nil)
      (ros::ros-info "[main] Recognizing objects in bin ~a" target-bin-)
      (unless (> start-picking-fail-count- 0)
        (send *ri* :speak (format nil "~a arm is going to bin ~a." (arm2str arm) target-bin-))
        (send self :move-arm-body->bin-overlook-pose arm target-bin-)
        (send *ri* :wait-interpolation))
      (setq recognition-count 1)
      (let ((stamp (ros::time-now)) order)
        (while (null (or (> recognition-count trial-times) is-recognized))
          (setq order (send self :select-work-order-in-bin arm target-bin- :stamp stamp))
          (setq is-recognized
            (when order
              (ros::ros-info-green "[main] arm: ~a, tried directions: ~a" arm tried-directions-)
              (when (null tried-directions-)
                (setq target-obj- (send order :item))
                (setq target-box- (str2symbol (send order :box))))
              (ros::set-param
                (format nil "~a_hand/target_object" (arm2str arm)) target-obj-)
              (ros::set-dynparam
                (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
                (cons "label_value" (position target-obj- label-names- :test #'string=)))
              (ros::ros-info-blue "[~a] [main] :recognize-object, target-bin: ~a, target-obj: ~a" (ros::get-name) target-bin- target-obj-)
              (ros::service-call
                (format nil "/~a_hand_camera/image_buffer/save" (arm2str arm))
                (instance std_srvs::TriggerRequest :init))
              (send self :recognize-target-object arm :stamp (ros::time-now))))
          (incf recognition-count)))
      (unless is-recognized
        (setq recognition-count 1)
        (let ((stamp (ros::time-now)) target-index)
          (while (null (or (> recognition-count trial-times) is-recognized))
            (setq target-index
                  (send self :get-largest-object-index arm target-bin-
                        :stamp stamp :n-random nil))
            (setq is-recognized
              (when target-index
                (setq target-obj- (elt label-names- target-index))
                (ros::set-dynparam
                  (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
                  (cons "label_value" target-index))
                (ros::set-param
                  (format nil "~a_hand/target_object" (arm2str arm)) target-obj-)
                (ros::ros-info-blue "[~a] [main] target recognition failed" (ros::get-name) target-bin-)
                (ros::ros-info-blue "[~a] [main] moving object in bin ~A" (ros::get-name) target-bin-)
                (ros::ros-info-blue "[~a] [main] :recognize-object, target-obj: ~a" (ros::get-name) target-obj-)
                (setq moving-object-p- t)
                (send self :recognize-target-object arm :stamp (ros::time-now))))
            (incf recognition-count))))
      is-recognized))
  (:return-from-recognize-object (arm)
    (send self :add-postponed-object arm target-obj- target-bin-)
    (setq start-picking-fail-count- 0)
    (ros::ros-info "[main] arm: ~a, failed to recognize object ~a" arm target-obj-)
    (send *ri* :angle-vector-sequence
          (list (send *baxter* :fold-to-keep-object arm)
                (send *baxter* :arc-reset-pose arm))
          :fast (send *ri* :get-arm-controller arm) 0 :scale 3.0)
    (send *ri* :wait-interpolation))
  (:check-recognize-fail-count (arm &key (count-limit 2))
    (incf recognize-fail-count-)
    (ros::ros-info "[main] arm: ~a, recognize fail count: ~a" arm recognize-fail-count-)
    (let (recognize-failp)
      (setq recognize-failp (null (> count-limit recognize-fail-count-)))
      (if recognize-failp (setq recognize-fail-count- 0))
      recognize-failp))
  (:check-can-start-picking (arm)
    ;; first trial: trial-time = 1
    (setq grasp-style- (send self :get-grasp-style arm target-obj- 1))
    (ros::ros-info-green "arm: ~a, grasp-style: ~a, trial-time: ~a"
                         (arm2str arm) (symbol2str grasp-style-) 1)
    (when grasp-style-
      (setq prev-target-obj- nil)
      (setq start-picking-fail-count- 0))
    (if grasp-style- t nil))
  (:check-start-picking-fail-count (arm &key (count-limit 2))
    (incf start-picking-fail-count-)
    (ros::ros-info "[main] arm: ~a, start picking fail count: ~a" arm start-picking-fail-count-)
    (send self :add-postponed-object arm target-obj- target-bin-)
    (when (and (> start-picking-fail-count- 1)
               (string= prev-target-obj- target-obj-))
      (setq prev-target-obj- nil)
      (return-from :check-start-picking-fail-count t))
    (setq prev-target-obj- target-obj-)
    (let (start-picking-failp)
      (setq start-picking-failp (null (> count-limit start-picking-fail-count-)))
      (when start-picking-failp
        (setq prev-target-obj- nil)
        (setq start-picking-fail-count- 0))
      start-picking-failp))
  (:check-tried-directions (arm)
    (send *ri* :stop-grasp arm)
    (ros::ros-info "[main] arm: ~a, tried directions: ~a" arm tried-directions-)
    (if (>= (length tried-directions-) 4)
      (progn
        (setq tried-directions- nil)
        nil)
      t))
  (:pick-object
    (arm)
    (send *ri* :speak
          (format nil "~a arm is picking ~a, which is~a an ordered object."
                  (arm2str arm)
                  (underscore-to-space target-obj-)
                  (if moving-object-p- " not" "")))
    (let (object-index obj-coords pick-result graspingp)
      ;; slots reset
      (setq start-picking-fail-count- 0)
      (setq recognize-fail-count- 0)
      (when moveit-p- (send self :delete-bin-scene target-bin-))
      (if (eq grasp-style- :suction)
        (send self :set-movable-region-for-bin :offset (list 35 35 0))
        (send self :set-movable-region-for-bin :offset (list 35 35 0)))
      (send *baxter* :head_pan :joint-angle (if (eq arm :larm) -70 70))
      ;; pick largest bounding box
      (setq object-index 0)
      (setq obj-coords
            (send self :get-object-coords arm
                  :object-index object-index :tried-directions tried-directions-))
      (if obj-coords
        (progn
          (pushback (send obj-coords :rotate-vector #f(1 0 0)) tried-directions-))
        (progn
          (ros::ros-error "[main] arm: ~a set dummy tried directions" arm)
          (setq tried-directions- (list t t t t))
          (ros::ros-error "[main] arm: ~a abort picking" arm)
          (return-from :pick-object nil)))
      (setq pick-result
            (send self :pick-object-in-bin arm target-bin- obj-coords
                  :n-trial 1
                  :n-trial-same-pos 1
                  :do-stop-grasp nil
                  :grasp-style grasp-style-))
      (when moveit-p- (send self :add-bin-scene target-bin-))
      (send self :ik->bin-center arm target-bin-
            :offset #f(0 0 300)
            :rpy (float-vector (if (eq arm :larm) -pi/2 pi/2) 0 0)
            :rotation-axis t
            :use-gripper nil
            :use-rotate-wrist-ik t)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000
            (send *ri* :get-arm-controller arm :gripper nil) 0)
      (send *ri* :wait-interpolation)
      (if (or (eq pick-result :ik-failed) (eq pick-result :plan-failed))
        (return-from :pick-object nil))
      ;; Trust pressure sensor
      (setq graspingp (send *ri* :graspingp arm :suction))
      (ros::ros-info "[main] arm: ~a graspingp: ~a" arm graspingp)
      ;; TODO (send *ri* :graspingp arm :pinch) always returns t,
      ;;      so placing always occurs even if nothing is pinched
      graspingp))
  (:return-from-pick-object (arm)
    (send self :add-postponed-object arm target-obj- target-bin-)
    (setq start-picking-fail-count- 0)
    (send *ri* :angle-vector-sequence-raw
          (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))
                (send *baxter* :arc-reset-pose arm))
          :fast (send *ri* :get-arm-controller arm)
          0 :scale 3.0)
    (send *ri* :wait-interpolation)
    (if (eq grasp-style- :pinch)
      (send self :resolve-collision-between-fingers arm))
    (when (eq grasp-style- :edge)
      (send *ri* :move-hand arm
          (send *baxter* :hand-grasp-pre-pose arm :cylindrical) 1000)
      (send *ri* :move-hand arm
          (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000)))
  (:verify-object (arm)
    (when moving-object-p-
      (setq moving-object-p- nil)
      (ros::ros-info "moving object in bin")
      (return-from :verify-object nil))
    ;; skip verify because we have neither external camera or scale
    t)
  (:check-picking-fail-count (arm &key (count-limit 2))
    (incf picking-fail-count-)
    (ros::ros-info "[main] arm: ~a, picking fail count: ~a" arm picking-fail-count-)
    (let (picking-failp)
      (setq picking-failp (null (> count-limit picking-fail-count-)))
      (if picking-failp (setq picking-fail-count- 0))
      picking-failp))
  (:set-place-target (arm)
    (setq picking-fail-count- 0)
    (if (eq target-box- :shelf)
      (progn
        (ros::ros-info "[main] blacklisted object, return to shelf")
        nil)
      (progn
        (setq place-target-label- (format nil "cardboard_~A" (symbol-string target-box-)))
        (setq place-target-cube- (gethash target-box- cardboard-cubes-))
        ;; logging
        (ros::ros-info "[main] place-target: label: ~a, cube: ~a"
                       place-target-label- place-target-cube-)
        t)))
  (:return-object (arm)
    (send self :add-postponed-object arm target-obj- target-bin-)
    (when moveit-p- (send self :delete-bin-scene target-bin-))
    (send *ri* :angle-vector-raw
          (send self :ik->bin-center arm target-bin-
                :offset (send self :decide-bin-return-area arm target-bin-)
                :rotation-axis :z :use-gripper t)
          :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp arm)
    (send *ri* :stop-grasp arm :pinch)
    (send self :spin-off-by-wrist arm :times 3)
    (send *ri* :wait-interpolation)
    (ros::ros-info "[main] ~a, return object in shelf" arm)
    (send *ri* :angle-vector-raw
          (send *baxter* arm :move-end-pos #f(0 0 200) :local :move-palm-end t :rotation-axis :z)
          :fast (send *ri* :get-arm-controller arm) 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    (when moveit-p- (send self :add-bin-scene target-bin-))
    (send self :arc-reset-pose arm)
    (send *ri* :wait-interpolation)
    (if (eq grasp-style- :pinch)
      (send self :resolve-collision-between-fingers arm))
    (when (eq grasp-style- :edge)
      (send *ri* :move-hand arm
          (send *baxter* :hand-grasp-pre-pose arm :cylindrical) 1000)
      (send *ri* :move-hand arm
          (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000)))
  (:place-object (arm)
    (let (dropped end-coords-name orientation path-constraints orientation-constraint place-z)
      ;; FIXME: moveit cannot solve path
      ;; moveit path constraints
      ;; (setq end-coords-name
      ;;       (if (eq arm :larm) "left_gripper_pad_with_base" "right_gripper_pad_with_base"))
      ;; (setq orientation (instance geometry_msgs::Quaternion :init :x 0 :y 0 :z 0 :w 1))
      ;; (setq orientation-constraint
      ;;       (instance moveit_msgs::OrientationConstraint :init
      ;;                 :link_name end-coords-name
      ;;                 :orientation orientation))
      ;; (send orientation-constraint :header :frame_id end-coords-name)
      ;; (setq path-constraints (instance moveit_msgs::Constraints :init
      ;;                                  :orientation_constraints (list orientation-constraint)))

      (send *ri* :speak
            (format nil "~a arm is placing ~a into ~a."
                    (arm2str arm) (underscore-to-space target-obj-)
                    (underscore-to-space place-target-label-)))
      (ros::ros-info "[main] ~a, place object in ~a" arm place-target-label-)
      ;; (when moveit-p- (send self :add-object-in-hand-scene arm target-obj-))

      ;; FIXME: moveit cannot solve path
      ;; (send *ri* :angle-vector
      ;;       (send self :ik->cardboard-center arm target-cardboard-
      ;;             :offset #f(0 0 200) :rotation-axis :z :use-gripper t)
      ;;       4000 (send *ri* :get-arm-controller arm) 0
      ;;       :path-constraints path-constraints)
      (setq tried-directions- nil)
      (let (avs (offsets (list #f(-50 0 500) #f(-50 0 500) #f(-50 0 500) #f(-50 0 250)))
                (rpys (list (float-vector pi/2 0 -pi/2) (float-vector pi/2 0 (/ (* pi 2) -3))
                            (float-vector pi/2 0 -pi/2) (float-vector pi/2 0 -pi/2))))
        (dotimes (i (length offsets))
          (pushback
            (send self :ik->cube-center arm place-target-cube-
                  :offset (elt offsets i)
                  :rpy (elt rpys i)
                  :rotation-axis t
                  :use-gripper nil
                  :use-rotate-wrist-ik t)
            avs))
        (setq avs (cdr avs))
        (setq avs (remove nil avs))
        (setq place-z 180)
        (send *ri* :angle-vector-sequence avs :fast
              (send *ri* :get-arm-controller arm :gripper nil :head t) 0 :scale 5.0)
        (send *ri* :wait-interpolation))
      (setq dropped (not (send *ri* :graspingp arm :suction)))
      ;; (when moveit-p- (send self :delete-object-in-hand-scene arm))
      (if dropped
        (progn
          (send *ri* :speak
                (format nil "~a arm dropped ~a to somewhere."
                        (arm2str arm) (underscore-to-space target-obj-)))
          (ros::ros-error "[main] arm ~a: dropped object" arm)
          (send *ri* :stop-grasp arm)
          (send *ri* :stop-grasp arm :pinch)
          (send self :add-postponed-object arm target-obj- target-bin-))
        (let (av)
          (ros::ros-info-green "[main] arm ~a: place object ~a in ~a"
                               arm target-obj- place-target-label-)
          ;; TODO: fix dummy updating of json
          ;; (send self :update-json target-obj-
          ;;       :src (cons :bin target-bin-) :dst place-target-label-)
          (send self :update-json target-obj-
                :src (cons :bin target-bin-) :dst (cons :cardboard target-box-))
          (send self :add-finished-object arm target-obj- target-bin-)
          (when moveit-p- (send *co* :delete-object place-target-cube-))
          (setq av (send *baxter* arm :move-end-pos (float-vector 0 0 (- place-z))
                         (send place-target-cube- :worldcoords)))
          (when av
            (send *ri* :angle-vector-raw av
                  2000 (send *ri* :get-arm-controller arm) 0)
            (send *ri* :wait-interpolation))
          ;; push object to container
          (send *ri* :angle-vector-raw
                (send *baxter* arm :move-end-pos (float-vector 0 70 0)
                      (send place-target-cube- :worldcoords)
                      :rotation-axis t)
                1000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (send *ri* :stop-grasp arm) ;; release object
          (send *ri* :stop-grasp arm :pinch)
          (unix::sleep 2)
          (send *ri* :angle-vector-raw
                (send *baxter* arm :move-end-pos (float-vector 10 0 0)
                      (send place-target-cube- :worldcoords)
                      :rotation-axis t)
                1000 (send *ri* :get-arm-controller arm) 0)
          (send *ri* :wait-interpolation)
          (when av
            (setq av (send *baxter* arm :move-end-pos (float-vector 0 0 place-z)
                           (send place-target-cube- :worldcoords)
                           :rotation-axis t))
            (send *ri* :angle-vector-raw av
                  1000 (send *ri* :get-arm-controller arm) 0)
            (send *ri* :wait-interpolation))
          (when moveit-p- (send self :add-labeled-obj-scene
                                place-target-cube- place-target-label-))))))
  (:return-from-place-object (arm)
    (let (avs)
      (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
      (pushback (send *baxter* :arc-reset-pose arm) avs)
      (send *ri* :angle-vector-sequence avs 5000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      (if (eq grasp-style- :pinch)
        (send self :resolve-collision-between-fingers arm))
      (when (eq grasp-style- :edge)
        (send *ri* :move-hand arm
              (send *baxter* :hand-grasp-pre-pose arm :cylindrical) 1000)
        (send *ri* :move-hand arm
              (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000))
      (setq grasp-style- nil))))


(defun grasp_prediction_arc2017::pick-book-init (&key (ctype :default-controller) (moveit nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance jsk_arc2017_baxter::baxterlgv7-moveit-environment))
      (setq mvit-rb (instance jsk_arc2017_baxter::baxterlgv7-robot :init)))
    (unless (boundp '*ri*)
      (setq *ri* (instance jsk_arc2017_baxter::baxterlgv7-interface :init :type ctype
                           :moveit-environment mvit-env
                           :moveit-robot mvit-rb)))
    (unless (boundp '*baxter*)
      (setq *baxter* (instance jsk_arc2017_baxter::baxterlgv7-robot :init)))
    (unless (boundp '*co*)
      (setq *co* (when moveit (instance collision-object-publisher :init))))
    (unless (boundp '*ti*)
      (setq *ti* (instance grasp_prediction_arc2017::pick-book-interface :init :moveit moveit)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)))
