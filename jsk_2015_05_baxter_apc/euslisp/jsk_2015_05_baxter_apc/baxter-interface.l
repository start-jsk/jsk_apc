;; -*- mode: lisp;-*-

(require "package://baxtereus/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")

(ros::load-ros-manifest "jsk_2015_05_baxter_apc")

(unless (find-package "JSK_2015_05_BAXTER_APC")
  (make-package "JSK_2015_05_BAXTER_APC"))

(defclass jsk_2015_05_baxter_apc::baxter-interface
  :super baxter-interface
  :slots
  (
   _tfl
   _bin-boxes
   _objects-in-bin-boxes
   )
  )

(defmethod jsk_2015_05_baxter_apc::baxter-interface
  (:init
    (&rest args)
    (send-super* :init args)
    ;; change robot model in robot-interface class
    (send (send robot :rarm :end-coords)
          :newcoords (make-coords :pos #f(50 0 310)
                                  :rpy (float-vector 0 -pi/2 0)))
    (send (send robot :larm :end-coords)
          :newcoords (make-coords :pos #f(50 0 310)
                                  :rpy (float-vector 0 -pi/2 0)))
    ;; initialize slots
    (setq _tfl (instance ros::transform-listener :init))
    (setq _bin-boxes (make-hash-table))
    (setq _objects-in-bin-boxes (make-hash-table))
    (unless (ros::get-param "~gazebo" nil)
      (ros::advertise "/vacuum_gripper/limb/left" std_msgs::Bool)
      (ros::advertise "/vacuum_gripper/limb/right" std_msgs::Bool)
      )
    )
  (:start-grasp
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (send self :arm-symbol2str arm))))
      (if (ros::get-param "~gazebo" nil)
        (ros::service-call (format nil "/robot/~a_vacuum_gripper/on" l/r) (instance std_srvs::EmptyRequest :init))
        (ros::publish (format nil "/vacuum_gripper/limb/~a" l/r) (instance std_msgs::Bool :init :data t))
        )
      ))
  (:stop-grasp
    (&optional (arm :arms))
    (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (send self :arm-symbol2str arm))))
      (if (ros::get-param "~gazebo" nil)
        (ros::service-call (format nil "/robot/~a_vacuum_gripper/off" l/r) (instance std_srvs::EmptyRequest :init))
        (ros::publish (format nil "/vacuum_gripper/limb/~a" l/r) (instance std_msgs::Bool :init :data nil))
        )
      ))
  (:graspingp
    (arm)
    (let (topic)
      (if (ros::get-param "~garebo" nil)
        (setq topic (format nil "/robot/~a_vacuum_gripper/grasping" (arm-to-str arm)))
        (setq topic (format nil "gripper_grabbed/limb/~a/state" (arm-to-str arm)))
        )
      (send (one-shot-subscribe topic std_msgs::Bool) :data)))
  (:opposite-arm (arm) (if (eq arm :larm) :rarm :larm))
  (:need-to-wait-opposite-arm
    (arm)
    (let (opposite-state)
      (setq opposite-state (str2symbol (ros::get-param (format nil "~a_hand/state" (send self :arm-symbol2str (send self :opposite-arm arm))))))
      (setq opposite-target (str2symbol (ros::get-param (format nil "~a_hand/target_bin" (send self :arm-symbol2str (send self :opposite-arm arm))))))
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
      (:larm (subseq (send *ri* :state :potentio-vector) 1 8))
      (:rarm (subseq (send *ri* :state :potentio-vector) 8 15))
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
  (:recognize-bin-boxes
    (&key (stamp (ros::time-now)))
    (let (msg bin bin-box-pos-z)
      ;; for bin A-F
      (setq
        msg
        (one-shot-subscribe "bin_atof_clipper/output/box_array"
                            jsk_recognition_msgs::BoundingBoxArray
                            :after-stamp stamp))
      (dotimes (i 6)
        (setq bin (elt '(:a :b :c :d :e :f) i))
        (setf (gethash bin _bin-boxes) (elt (send msg :boxes) i)))
      ;; for bin G-L
      (setq
        msg
        (one-shot-subscribe "bin_gtol_clipper/output/box_array"
                            jsk_recognition_msgs::BoundingBoxArray
                            :after-stamp stamp))
      (dotimes (i 6)
        (setq bin (elt '(:j :k :l :g :h :i) i))
        (setf (gethash bin _bin-boxes) (elt (send msg :boxes) i)))
      ))
  (:bbox->cube
    (bbox)
    (let* ((dims (ros::tf-point->pos (send bbox :dimensions)))
           (bx (make-cube (elt dims 0) (elt dims 1) (elt dims 2))))
      (send bx :newcoords
            (send self :tf-pose->coords
                  (send bbox :header :frame_id)
                  (send bbox :pose)))
      bx))
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
  (:recognize-objects-in-bin
    (bin &key (stamp (ros::time-now)))
    (setq
      msg
      (one-shot-subscribe
        (format nil "bin_~a_cpi_decomposer/boxes" (symbol2str bin))
        jsk_recognition_msgs::BoundingBoxArray
        :after-stamp stamp))
    (sethash bin _objects-in-bin-boxes (send msg :boxes)))
  (:recognize-object-in-hand
    (arm &key (stamp (ros::time-now)))
    (let* (msg)
      (setq msg
            (one-shot-subscribe
              (format nil "~a_hand/object_verification/output" (arm-to-str arm))
              jsk_2015_05_baxter_apc::ObjectRecognition
              :after-stamp stamp))
      (if msg
        (progn
          (ros::ros-info "[:recognize-object-in-hand] recognized as: ~a" (send msg :matched))
          (send msg :matched))
        nil)
      ))
  (:verify-object
    (arm object-name &key (stamp (ros::time-now)))
    (let (matched)
      (if (setq matched (send self :recognize-object-in-hand arm :stamp stamp))
        (progn (ros::ros-info "~a == ~a" object-name matched)
               (string= object-name matched))
        nil)
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
  (:try-to-pick-object
    (arm bin &key (object-index 0) (offset #f(0 0 0)))
    (let (avs obj-boxes obj-box)
      ;; validate
      (unless
        (setq obj-boxes (gethash bin _objects-in-bin-boxes))
        (ros::ros-warn "No object is found: ~a, ~a" arm bin)
        (return-from :try-to-pick-object nil))
      (unless
        (setq obj-box (elt obj-boxes object-index))
        (ros::ros-warn "The object is not found: ~a ~a ~a" arm bin object-index)
        (return-from :try-to-pick-object nil))
      ;; ik to obj a bit distant
      (setq obj-coords (send self :tf-pose->coords (send obj-box :header :frame_id) (send obj-box :pose)))
      (if (find bin '(:a :b :c :d :e :f))
        (setq world-x :z world-y :x world-z :y)
        (setq world-x :z world-y :y world-z :x)
        )
      (if (and
            (< (send obj-box :dimensions world-y) (send obj-box :dimensions world-z))
            (> (send obj-box :dimensions world-z) (/ (send (gethash bin _bin-boxes) :dimensions :z) 2))
            )
        (setq sign (if (< (elt (send obj-coords :pos) 1) (elt (send (send *baxter* arm :end-coords) :worldpos) 1)) -1 1)) ;; compare y
        (setq sign 0))
      (ros::ros-info "[:try-to-pick-object] sign: ~a" sign)
      (send self :angle-vector
            (send self :ik->bin-entrance arm bin :offset (float-vector -30 0 (m->mm (/ (send (gethash bin _bin-boxes) :dimensions :z) 4))))
            3000)
      (send self :wait-interpolation)
      ; (pushback
      ;   (send *baxter* arm :inverse-kinematics
      ;       (send (send (send *baxter* arm :end-coords) :copy-worldcoords) :translate offset :local))
      ;   avs)
      (pushback
        (send *baxter* arm :inverse-kinematics
            (send (send (send *baxter* arm :end-coords) :copy-worldcoords) :rotate (* sign pi/2) :x :local))
        avs)
      (send self :angle-vector-sequence avs :fast nil 0 :scale 5.0)
      (send self :wait-interpolation)
      ;; start the vacuum gripper
      (send self :start-grasp arm)
      ;; grasp object
      (send self :angle-vector
            (send *baxter* arm :inverse-kinematics
                  (make-coords :pos (send obj-coords :pos) :rpy (list 0 0 (* sign pi/2))))
            3000)
      (send self :wait-interpolation)
      (unless (send self :graspingp arm)
        (send self :angle-vector (send *baxter* arm :move-end-pos #f(0 0 -50) :local) 3000)
        (send self :wait-interpolation))
      ;; lift object
      (send self :angle-vector (send *baxter* arm :move-end-pos #f(0 0 80) :local) 3000)
      (send self :wait-interpolation)
    ))
  (:pick-object
    (arm bin &key (object-index 0) (n-trial 1) (n-trial-same-pos 1))
    (send *ri* :angle-vector-sequence
      (list (send self :ik->bin-entrance arm bin :offset #f(-100 0 50)))
      :fast nil 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    (let (bin-box)
      ;; move arm bin-entrance -> pos above object to z direction
      (setq bin-box (gethash bin _bin-boxes))
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (unless
            ;; try to pick and the goal becomes far to x gradually
            (send self :try-to-pick-object arm bin :object-index object-index :offset (float-vector (+ 0 (* i 20)) 0 (m->mm (/ (send bin-box :dimensions :z) 2))))
            (ros::ros-warn "No objects found, try to pick without vision result")
            (send self :try-to-pick-in-bin arm bin))
          (when (and (< (send self :real-sim-end-coords-diff arm) 60) (send self :graspingp arm)) (return)) ;; not wall and grasped
          (send self :stop-grasp arm)
          )
        (when (and (< (send self :real-sim-end-coords-diff arm) 60) (send self :graspingp arm)) (return)) ;; not wall and grasped
        (send self :stop-grasp arm)
        )
      (send self :angle-vector (send self :ik->bin-entrance arm bin) 3000)
      (send self :wait-interpolation)
      ;; move arm in-bin -> bin-entrance
      (send self :angle-vector-sequence
        (list
          (send *baxter* arm :move-end-pos #f(0 0 50))
          (send self :ik->bin-entrance arm bin :offset #f(0 0 50))
          (send self :ik->bin-entrance arm bin :offset #f(-100 0 50))
          (send self :ik->bin-entrance arm bin :offset #f(-300 0 50))
          )
        :fast nil 0 :scale 5.0)
      (send self :wait-interpolation)
      ;; move arm bin-entrance -> body
      (send *baxter* :avoid-shelf-pose arm bin)
      (send self :send-av)
      (send self :wait-interpolation)
      ))
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
          (> (norm (v- (send *baxter* arm :angle-vector) (send *ri* :arm-potentio-vector arm))) threshold)
          (when (> times stop) (ros::ros-warn "abort to send avs: ~A" arm) (return))  ;; abort
          (ros::ros-info "difference of avs: ~A" (norm (v- (send *baxter* arm :angle-vector) (send *ri* :arm-potentio-vector arm))))
          (send *ri* :angle-vector (send *baxter* :angle-vector))
          (send *ri* :wait-interpolation)
          (incf times)
          )
        ))
    )
  (:ik->bin-entrance
    (arm bin &key (offset #f(0 0 0)))
    (let (bin-box bin-coords bin-dim-x)
      (setq bin-box (gethash bin _bin-boxes))
      (unless bin-box
        (ros::ros-error "[:ik->bin-entrance] No data about bin-box ~a. Call :recognize-bin-boxes first." bin)
        (return-from :ik->bin-entrance))
      (setq bin-coords (send self :tf-pose->coords
                             (send (send bin-box :header) :frame_id)
                             (send bin-box :pose)))
      (setq bin-dim-x (m->mm (send (send bin-box :dimensions) :x)))
      (send bin-coords :translate (float-vector (- (/ bin-dim-x 2)) 0 0) :world)
      (case bin
        ((:a :b :c) (send *baxter* :fold-pose-upper arm))
        ((:e :j :k :l) (send *baxter* :fold-pose-middle arm))
        ((:d :f :g :h :i) (send *baxter* :fold-pose-lower arm))
        )
      (send *baxter* arm :inverse-kinematics bin-coords :rotation-axis t)
      ;; apply offset
      (send bin-coords :translate offset :world)
      (send *baxter* arm :inverse-kinematics bin-coords
            :rotation-axis :z
            :revert-if-fail nil)))
  (:move-arm-body->bin
    (arm bin)
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :avoid-shelf-pose arm bin) avs)
      (pushback (send self :ik->bin-entrance arm bin :offset #f(-150 0 0)) avs)
      (send self :angle-vector-sequence avs :fast nil 0 :scale 3.0)
      ))
  (:move-arm-body->order-bin
    (arm)
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :place-object-pose arm) avs)
      (send self :angle-vector-sequence avs)
      ))
  (:spin-off-by-wrist
    (arm &key (times 10))
    (let (avs)
      (pushback (send *baxter* :angle-vector) avs)
      (dotimes (i 10)
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
      (send self :angle-vector-sequence avs)
      ))
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
      (when (null current-order) (return-from :get-next-work-order (elt orders 0)))
      (dotimes (i (- (length orders) 1))
        (when (string= (send (elt orders i) :bin) (send current-order :bin))
          (return-from :get-next-work-order (elt orders (+ i 1)))))))
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
      (setq real-coords (send (send robot arm :root-link) :transformation (send robot arm :end-coords)))
      (setq sim-coords (send (send *baxter* arm :root-link) :transformation (send *baxter* arm :end-coords)))
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
        (setq can-start (send (ros::service-call "rqt_yn_btn" (instance jsk_rqt_plugins::YesNoRequest)) :yes))
        )
      (ros::ros-info "[:wait-for-user-input-to-start] received user input: ~a" arm)
      ))
  )

(defun jsk_2015_05_baxter_apc::baxter-init (&key (ctype :default-controller))
  (unless (boundp '*ri*)
    (setq *ri* (instance jsk_2015_05_baxter_apc::baxter-interface :init :type ctype)))
  (unless (boundp '*baxter*)
    (setq *baxter* (instance jsk_2015_05_baxter_apc::baxter-robot :init)))
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *ri* :calib-grasp :arms)
  )
