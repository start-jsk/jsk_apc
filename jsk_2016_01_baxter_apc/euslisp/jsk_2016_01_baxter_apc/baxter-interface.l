;; -*- mode: lisp;-*-

(require "package://baxtereus/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter.l")

(ros::load-ros-manifest "jsk_2016_01_baxter_apc")

(unless (find-package "JSK_2016_01_BAXTER_APC")
  (make-package "JSK_2016_01_BAXTER_APC"))

(defclass jsk_2016_01_baxter_apc::baxter-interface
  :super baxter-interface
  :slots
  (
   _tfl
   _bin-boxes
   _objects-in-bin-boxes
   _objects-in-bin-coms
   _objects-sib-boxes
   _objects-sib-coords
   _bin-coords-list
   _hard-coded-variables
   _last-av-before-approach
   )
  )

(defmethod jsk_2016_01_baxter_apc::baxter-interface
  (:init
    (&rest args)
    ;; initialize controllers
    (send-super* :init :joint-states-queue-size 2 args)
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
    (setq _tfl (instance ros::transform-listener :init))
    (setq _bin-boxes (make-hash-table))
    (setq _objects-in-bin-boxes (make-hash-table))
    (setq _objects-in-bin-coms (make-hash-table))
    (setq _objects-sib-boxes (make-hash-table))
    (setq _objects-sib-coords (make-hash-table))
    ;; set hard-coded variables
    (setq _hard-coded-variables (make-hash-table))
    (sethash :r-pressure-threshold _hard-coded-variables 1000)
    (sethash :l-pressure-threshold _hard-coded-variables 1000)
    (sethash :offset-avoid-bin-top _hard-coded-variables -20)
    (sethash :pad-link-l _hard-coded-variables 85)
    (sethash :gripper-tube-t _hard-coded-variables 40)
    (sethash :offset-object-h _hard-coded-variables -10)
    (sethash :offset-gripper-bin-top _hard-coded-variables 10)
    (sethash :offset-gripper-bin-side _hard-coded-variables 40)
    (sethash :draw-out-from-bin-l _hard-coded-variables 80)

    (unless (ros::get-param "/apc_on_gazebo" nil)
      (ros::advertise "/vacuum_gripper/limb/left" std_msgs::Bool)
      (ros::advertise "/vacuum_gripper/limb/right" std_msgs::Bool)
      )
    (ros::advertise "/gripper_front/limb/left/servo/torque" std_msgs::Bool)
    (ros::advertise "/gripper_front/limb/right/servo/torque" std_msgs::Bool)
    )

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
                (gethash :r-pressure-threshold _hard-coded-variables))
               ((eq arm :larm)
                (gethash :l-pressure-threshold _hard-coded-variables))
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
      (send (send _tfl :lookup-transform "base" frame_id (ros::time 0)) :transform coords)
      ))
  (:fold-pose-back
    (&optional (arm :arms))
    (send *baxter* :fold-pose-back arm)
    (send *ri* :send-av))
  (:detect-target-object-in-bin
    (target-object bin)
    (let (obj-sizes box-sizes obj-idx box-idx box-size box-sizes-back
                    (candidates (send self :get-bin-contents bin))
                    (boxes (gethash bin _objects-in-bin-boxes)))
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
    (let ((box-topic (format nil "publish_bin_bbox/boxes"))
          box-msg bin-list)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[:recognize-bin-boxes] recognize bin boxes")
          (setq box-list (send box-msg :boxes))
          (setq bin-list (list :a :b :c :d :e :f :g :h :i :j :k :l))
          (dolist (bin bin-list)
            (setf (gethash bin _bin-boxes) (car box-list))
            (setq box-list (cdr box-list))))
        (ros::ros-fatal "[:recognize-bin-boxes] cannot recognize bin boxes"))))
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
  (:visualize-bins
    ()
    (let (bins)
      (dolist (b (send _bin-boxes :list-values))
        (pushback (send self :bbox->cube b) bins)
        )
      (objects (append (list *baxter*) bins))
      ))
  (:visualize-objects
    ()
    (let (objs)
      (dolist (boxes (send _objects-in-bin-boxes :list-values))
        (dolist (b boxes)
          (pushback (send self :bbox->cube b) objs)
          ))
      (objects (append (list *baxter*) objs))
      ))
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
      (setq _bin-coords-list coords-list)
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
      (sethash bin _objects-in-bin-boxes (send box-msg :boxes))
      (sethash bin _objects-in-bin-coms com-msg)))
  (:recognize-objects-segmentation-in-bin
    (arm bin &key (stamp (ros::time-now)) (timeout 30))
    (let ((box-topic (format nil "~a_sib_cpi_decomposer/boxes" (send self :arm-symbol2str arm)))
          box-msg
          (com-topic (format nil
                             "~a_sib_cpi_decomposer/centroid_pose_array"
                             (send self :arm-symbol2str arm)))
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
         (sethash bin _objects-sib-boxes obj-box)
         (sethash bin _objects-sib-coords obj-coords))
        (t
         (ros::ros-error "[:recognize-objects-segmentation-in-bin] arm: ~a failed to get sib msg" arm)))))
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
          (ros::ros-info "[:recognize-object-in-hand] recognized as: ~a" (send msg :matched))
          (send msg :matched))
        nil)
      ))
  (:verify-object
    (arm object-name &key (stamp (ros::time-now)))
    (let (matched)
      (if (setq matched (send self :recognize-object-in-hand arm :stamp stamp :timeout 10000))
        (progn (ros::ros-info "~a == ~a" object-name matched)
               (string= object-name matched))
        :timeout)
      ))
  (:try-to-pick-in-bin
    (arm bin)
    (let* ((bin-box (gethash bin _bin-boxes))
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
      (unless _bin-coords-list
        (ros::ros-error "no solidity coords-list is found")
        (return-from :try-to-pick-object-solidity))
      (dolist (c _bin-coords-list)
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
  (:get-obj-box-and-coords
    (arm bin &key (object-index 0) (use-sib t))
    (let (obj-box obj-coords obj-boxes obj-coms-msg obj-com)
      (cond (use-sib
              (unless
                (setq obj-box (elt (gethash bin _objects-sib-boxes) object-index))
                (ros::ros-warn "[SIB] No bbox  is found: ~a, ~a" arm bin)
                (return-from :get-obj-box-and-coords nil))
              ;; with Center of Mass
              (unless
                (setq obj-coords (elt (gethash bin _objects-sib-coords) object-index))
                (ros::ros-warn "[SIB] No com is found: ~a, ~a" arm bin)
                (return-from :get-obj-box-and-coords nil))
              )
            (t
              (unless
                (setq obj-boxes (gethash bin _objects-in-bin-boxes))
                (ros::ros-warn "No object is found: ~a, ~a" arm bin)
                (return-from :get-obj-box-and-coords nil))
              (unless
                (setq obj-box (elt obj-boxes object-index))
                (ros::ros-warn "The object is not found: ~a ~a ~a" arm bin object-index)
                (return-from :get-obj-box-and-coords nil))
              (setq obj-coms-msg (gethash bin _objects-in-bin-coms))
              (setq obj-com (elt (send obj-coms-msg :poses) object-index))
              ;; ik to obj a bit distant
              ;; with Center of Mass
              (setq obj-coords (send self :tf-pose->coords
                                     (send obj-coms-msg :header :frame_id) obj-com))
              )
            )
      (print obj-box)
      (print obj-coords)
      (print (list obj-box obj-coords)))
    )
  (:decide-approach-direction
    (obj-cube bin-cube)
    (let (sign sign_y gripper-req-l)
      ;; set requisite length for gripper to enter bin
      (setq gripper-req-l (+ (gethash :pad-link-l _hard-coded-variables)
                             (/ (gethash :gripper-tube-t _hard-coded-variables) 2)))
      ;; if object is higher than highest end-coords when gripper is 90 degrees
      (if (> (+ (elt (send obj-cube :pos) 2) (/ (z-of-cube obj-cube) 2))
             (- (+ (elt (send bin-cube :pos) 2) (/ (z-of-cube bin-cube) 2))
                gripper-req-l
                (gethash :offset-object-h _hard-coded-variables))
             )
        (cond ((< gripper-req-l
                  ;; if space gripper can enter exists on right
                  (- (- (elt (send obj-cube :pos) 1) (/ (y-of-cube obj-cube) 2))
                     (- (elt (send bin-cube :pos) 1) (/ (y-of-cube bin-cube) 2)))
                  )
               (setq sign 1)
               (setq sign_y 0))
              ((< gripper-req-l
                  ;; if space gripper can enter exists on left
                  (- (+ (elt (send bin-cube :pos) 1) (/ (y-of-cube bin-cube) 2))
                     (+ (elt (send obj-cube :pos) 1) (/ (y-of-cube obj-cube) 2)))
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
      (list sign sign_y))
    )
  (:try-to-pick-object
    (arm bin &key (object-index 0) (offset #f(0 0 0)) (use-sib t))
    (let (avs-before-approach obj-box graspingp gripper-req-l bin-box bin-y-l bin-z-l)
      ;; validate
      (let (obj-box-and-coords
             (send self :get-obj-box-and-coords
                   arm bin :object-index object-index :use-sib use-sib))
        (if (null obj-box-and-coords)
          (return-from :try-to-pick-object nil))
        (setq obj-box (elt obj-box-and-coords 0))
        (setq obj-coords (elt obj-box-and-coords 1)))
      ;; set requisite length for gripper to enter bin
      (setq gripper-req-l (+ (gethash :pad-link-l _hard-coded-variables)
                             (/ (gethash :gripper-tube-t _hard-coded-variables) 2)))
      (unless
        (setq bin-box (gethash bin _bin-boxes))
        (ros::ros-error "[:try-to-pick-object] No data about bin-box ~a." bin)
        (ros::ros-error "[:try-to-pick-object] Call :recognize-bin-boxes first.")
        (return-from :try-to-pick-object nil))
      (let (bin-cube-base obj-cube-base approach-direction)
        (setq bin-cube-base (send self :cube->cube-parallel-to-coords
                                  (send self :bbox->cube bin-box)
                                  :included-in-original-cube)
              obj-cube-base (send self :cube->cube-parallel-to-coords
                                  (send self :bbox->cube obj-box)
                                  :include-original-cube)
              bin-y-l (y-of-cube bin-cube-base)
              bin-z-l (z-of-cube bin-cube-base))
        (setq approach-direction (send self :decide-approach-direction obj-cube-base bin-cube-base))
        (setq sign (elt approach-direction 0))
        (setq sign_y (elt approach-direction 1))
        )
      (ros::ros-info "[:try-to-pick-object] arm:~a sign: ~a sign_y: ~a" arm sign sign_y)
      (if (ros::get-param "/apc_on_gazebo" nil)
        ;; FIXME: this condition is needed to avoid collision to the shelf at this moment
        (setq offset-from-entrance #f(-30 0 0))
        (if (< sign_y 0)
          (setq offset-from-entrance (float-vector -30 0 0))  ;; when gripper goes straight
          (setq offset-from-entrance
                (float-vector
                  (if (= sign 0) -30 (- (gethash :draw-out-from-bin-l _hard-coded-variables)))
                  0
                  (- (/ bin-z-l 2) gripper-req-l
                     (gethash :offset-gripper-bin-top _hard-coded-variables))
                  )
                )
          )
        )
      ;; initialize _last-av-before-approach
      (setq _last-av-before-approach (send *baxter* :angle-vector))
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
                  (- (+ (/ bin-dim-x 2) (gethash :draw-out-from-bin-l _hard-coded-variables)))
                  (* sign (- gripper-req-l (/ bin-y-l 2)
                             (- (gethash :offset-gripper-bin-side _hard-coded-variables))))
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
                    (gethash :draw-out-from-bin-l _hard-coded-variables)
                    0 0)
                  :world)
            avs-before-approach)
          (pushback
            (send *baxter* arm :move-end-pos
                  (float-vector
                    0 0
                    (+ (gethash :offset-gripper-bin-side _hard-coded-variables) 10)
                    )
                  :local)
            ;; press back of gripper against bin wall
            avs-before-approach)
          )
        )
      ;; check if IK to object succeed
      ;; if that IK fails, abort picking
      (if (send *baxter* arm :inverse-kinematics
                (make-coords :pos (send obj-coords :pos)
                             :rpy (list 0 (* sign_y pi/2) (* sign pi/2))))
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
          (setq _last-av-before-approach (send *baxter* :angle-vector))
          ;; grasp object
          (ros::ros-info "[:try-to-pick-object] arm:~a approach to the object" arm)
          (let (target-coords)
            (setq target-coords
                  (make-coords
                    :pos (v+ (send obj-coords :pos) offset)
                    :rpy (list 0 (* sign_y pi/2) (* sign pi/2))
                    ))
            (if (< sign_y 0)
              ;; when gripper is straight
              ;; move arm to the target only if robot can solve IK
              (if (send *baxter* arm :inverse-kinematics target-coords)
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
                (setq end->above (send end-coords :transformation target-coords :local))
                (setf (elt (send end->above :pos) 2) 0)
                (setq target-above-coords
                      (send end-coords :transform end->above :local))
                ;; move arm in the local x, y direction only if robot can solve IK
                (if (send *baxter* arm :inverse-kinematics target-above-coords
                          :revert-if-fail t)
                  (send self :angle-vector (send *baxter* :angle-vector)
                        (setq elapsed-t 2500))
                  )
                (send self :wait-interpolation)
                ;; move arm to the target
                ;; if above IK succeeded, arm moves in the local z direction
                (if (send *baxter* arm :inverse-kinematics target-coords)
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
            )
          ;; move arm to the target only if robot can solve IK
          (if (send *baxter* arm :inverse-kinematics
                    (make-coords :pos (send obj-coords :pos)
                                 :rpy (list 0 (* sign_y pi/2) (* sign pi/2))))
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
                     (if (send *baxter* arm :move-end-pos #f(0 0 20) :world)
                       (send self :angle-vector (send *baxter* :angle-vector) 3000)
                       (ros::ros-warn "[:try-to-pick-object] arm:~a fail to lift the object" arm)))
                    (t
                      ;; only if robot can solve IK
                      (if (send *baxter* arm :move-end-pos #f(0 0 60) :local)
                        (send self :angle-vector (send *baxter* :angle-vector) 3000)
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
          (send *baxter* :angle-vector _last-av-before-approach)  ;; revert *baxter*
          )
        )
      (setq graspingp (send self :graspingp arm))
      (ros::ros-info "[:try-to-pick-object] arm:~a graspingp: ~a" arm graspingp)
      graspingp))
  (:pick-object
    (arm bin &key (object-index 0) (n-trial 1) (n-trial-same-pos 1) (do-stop-grasp nil) (use-sib t))
    (send *ri* :angle-vector-sequence
          (list (send self :ik->bin-entrance arm bin
                      :offset (float-vector
                                -150 0
                                (gethash :offset-avoid-bin-top _hard-coded-variables)
                                )
                      ))
          :fast nil 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    (let (bin-box graspingp avs)
      ;; abort when no objects found
      (if use-sib
        (if
          (or
            (eq (length (gethash bin _objects-sib-boxes)) 0)
            (eq (length (gethash bin _objects-sib-coords)) 0))
          (return-from :pick-object nil) t)
        (if
          (eq (length (gethash bin _objects-in-bin-boxes)) 0)
          (return-from :pick-object nil) t)
        )
      ;; move arm bin-entrance -> pos above object to z direction
      (setq bin-box (gethash bin _bin-boxes))
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (unless graspingp
            (setq graspingp
                  (send self :try-to-pick-object arm bin :object-index object-index
                        :offset (float-vector (* i 50) 0 0) :use-sib use-sib)
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
      (send self :angle-vector-sequence
            (list
              (send *baxter* :angle-vector _last-av-before-approach)
              (send *baxter* arm :move-end-pos
                    (float-vector
                      (- (gethash :draw-out-from-bin-l _hard-coded-variables))
                      0 0)
                    :world)
              (send self :ik->bin-entrance arm bin
                    :offset (float-vector
                              -100 0 (gethash :offset-avoid-bin-top _hard-coded-variables))
                    )
              (send self :ik->bin-entrance arm bin
                    :offset (float-vector
                              -200 0 (gethash :offset-avoid-bin-top _hard-coded-variables))
                    )
              )
            :fast nil 0 :scale 5.0)
      (send self :wait-interpolation)
      ;; move arm bin-entrance -> body
      (send *baxter* :avoid-shelf-pose arm bin)
      (send self :send-av)
      (send self :wait-interpolation)
      (send self :gripper-servo-on arm)
      graspingp))
  (:send-av
    (&optional (tm 3000))
    (send self :angle-vector (send *baxter* :angle-vector) tm))
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
  (:ik->nearest-pose
    (arm target-coords pose-candidates &rest args)
    (if (null pose-candidates)
      (setq pose-candidates (list (send *baxter* :angle-vector))))
    (let (current-angle-vector ik-solvable-poses chosen-pose)
      (setq current-angle-vector (send *baxter* arm :angle-vector))
      ;; exclude poses which IK fail from
      (dolist (pose pose-candidates)
        (send *baxter* :angle-vector pose)
        (if (send* *baxter* arm :inverse-kinematics target-coords args)
          (pushback pose ik-solvable-poses)))
      (setq chosen-pose
            (car (sort ik-solvable-poses #'<
                       #'(lambda (pose)
                           (norm
                             (v-
                               current-angle-vector
                               (progn
                                 (send *baxter* :angle-vector pose)
                                 (send* *baxter* arm :inverse-kinematics target-coords args)
                                 (send *baxter* arm :angle-vector))))))))
      (format t "[:ik->nearest-pose] arm:~a midpose: ~a~%" arm chosen-pose)
      (send *baxter* :angle-vector chosen-pose)
      (send* *baxter* arm :inverse-kinematics target-coords args)
      )
    )
  (:ik->bin-entrance
    (arm bin &key (offset #f(0 0 0)) (gripper-angle 90)
         (rotation-axis :z) (overlook-angle 0) (revert-if-fail nil) (use-current-pose nil))
    (let (bin-box bin-coords bin-dim-x)
      (setq bin-box (gethash bin _bin-boxes))
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
      (send self :ik->nearest-pose arm bin-coords
            (list
              (send *baxter* :fold-pose-upper arm)
              (send *baxter* :fold-pose-middle arm)
              (send *baxter* :fold-pose-lower arm)
              (send *baxter* :avoid-shelf-pose arm bin)
              )
            :rotation-axis t
            )
      ;; apply rotation
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (send bin-coords :rotate (deg2rad (- gripper-angle 90)) :y :world)
      ;; apply offset
      (send bin-coords :translate offset :world)
      (send bin-coords :rotate overlook-angle :y :world)
      (if (or (< gripper-angle 45) (eq bin :h))
        (send *baxter* arm :inverse-kinematics bin-coords
              :rotation-axis t
              ;; if this is :z when gripper is straight, wrist sometimes rotates overly
              ;; if this is :z when Bin is h, body and arm collides on pulling out arm
              :revert-if-fail revert-if-fail)
        (send *baxter* arm :inverse-kinematics bin-coords
              :rotation-axis rotation-axis
              :revert-if-fail revert-if-fail)
        )
      ))
  (:move-arm-body->bin
    (arm bin)
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :avoid-shelf-pose arm bin) avs)
      (pushback (send self :ik->bin-entrance arm bin
                      :offset (float-vector
                                -150 0 (gethash :offset-avoid-bin-top _hard-coded-variables))
                      )
                avs)
      (send self :angle-vector-sequence avs :fast nil 0 :scale 3.0)
      ))
  (:move-arm-body->bin-overlook-pose
    (arm bin)
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :avoid-shelf-pose arm bin) avs)
      (pushback (send self :ik->bin-entrance arm bin
                      :offset #f(-150 0 -150)
                      :rotation-axis t
                      :gripper-angle 90
                      :overlook-angle (/ pi 9)) avs)
      (send self :angle-vector-sequence avs :fast nil 0 :scale 3.0)
      ))
  (:move-arm-body->order-bin
    (arm)
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :place-object-pose arm) avs)
      (send self :angle-vector-sequence avs :fast nil 0 :scale 3.0)
      ))
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
      (send self :angle-vector-sequence avs)
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
        (format nil "work_order/~a_hand" (send self :arm-symbol2str arm))
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
  (:check-bin-exist
    (bin)
    (if (gethash bin _bin-boxes) t nil))
  (:get-bin-contents
    (bin)
    (let (msg)
      (setq msg (one-shot-subscribe "bin_contents" jsk_2015_05_baxter_apc::BinContentsArray))
      (dolist (c (send msg :array))
        (when (string= (symbol2str bin) (send c :bin))
          (return-from :get-bin-contents (send c :objects))))))
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
      (ros::wait-for-service "rqt_yn_btn")
      (while
        (not can-start)
        (setq can-start (send (ros::service-call
                                "rqt_yn_btn" (instance jsk_rqt_plugins::YesNoRequest)) :yes))
        )
      (ros::ros-info "[:wait-for-user-input-to-start] received user input: ~a" arm)
      ))
  (:update-pressure-threshold
    (arm)
    (let (pressure hashkey topic)
      (cond
        ((eq arm :rarm)
         (setq hashkey :r-pressure-threshold))
        ((eq arm :larm)
         (setq hashkey :l-pressure-threshold)))
      (setq topic (format nil "/gripper_front/limb/~a/pressure/state" (arm-to-str arm)))
      (setf pressure (- (send (one-shot-subscribe topic std_msgs::Float64) :data) 7))
      (if (< pressure (gethash hashkey _hard-coded-variables))
        (setf (gethash hashkey _hard-coded-variables) pressure))))

  (:calib-pressure-threshold
    (&optional (arm :arms))
    (send self :start-grasp arm)
    (dotimes (i 7)
      (dolist (l/r (if (eq arm :arms) (list :rarm :larm) (list arm)))
        (send self :update-pressure-threshold l/r)
        )
      (unix::sleep 1))
    (send self :stop-grasp arm)
    (ros::ros-info "[:calib-pressure-threshold] Threshold r:~a l:~a"
                   (gethash :r-pressure-threshold _hard-coded-variables)
                   (gethash :l-pressure-threshold _hard-coded-variables)))

  )

(defun jsk_2016_01_baxter_apc::baxter-init (&key (ctype :default-controller))
  (unless (boundp '*ri*)
    (setq *ri* (instance jsk_2016_01_baxter_apc::baxter-interface :init :type ctype)))
  (unless (boundp '*baxter*)
    (setq *baxter* (instance jsk_2016_01_baxter_apc::baxter-robot :init)))
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *ri* :calib-grasp :arms)
  )
