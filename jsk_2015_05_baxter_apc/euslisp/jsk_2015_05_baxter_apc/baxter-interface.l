;; -*- mode: lisp;-*-

(require "package://baxtereus/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "jsk_recognition_msgs")

(unless (find-package "JSK_2015_05_BAXTER_APC")
  (make-package "JSK_2015_05_BAXTER_APC"))

(defclass jsk_2015_05_baxter_apc::baxter-interface
  :super baxter-interface
  :slots
  (
   tfl
   bin-atof-pos-stamp bin-boxes
   objects-in-bin-stamp objects-in-bin-boxes objects-in-bin-target
   work-order get-work-order-arm
   bin-contents
   )
  )

(defmethod jsk_2015_05_baxter_apc::baxter-interface
  (:init
    (&rest args)
    (send-super* :init args)
    (setq tfl (instance ros::transform-listener :init))
    (setq objects-in-bin-boxes (make-hash-table))
    (setq work-order (make-hash-table))
    )
  (:start-grasp
   (&optional (arm :arms))
   (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (send self :arm-symbol-to-str arm))))
     (ros::advertise-if-yet (format nil "/vacuum_gripper/limb/~A" l/r) std_msgs::Bool 1)
     (setq msg (instance std_msgs::Bool :init))
     (send msg :data t)
     (ros::ros-info (format nil "start ~A vacuum gripper" l/r))
     (ros::publish (format nil "/vacuum_gripper/limb/~A" l/r) msg)
     (ros::sleep)
     ))
  (:go-grasp (&optional (arm :arms)) (send self :start-grasp arm))
  (:stop-grasp
   (&optional (arm :arms))
   (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (send self :arm-symbol-to-str arm))))
     (ros::advertise-if-yet (format nil "/vacuum_gripper/limb/~A" l/r) std_msgs::Bool 1)
     (setq msg (instance std_msgs::Bool :init))
     (send msg :data nil)
     (ros::ros-info (format nil "stop ~A vacuum gripper" l/r))
     (ros::publish (format nil "/vacuum_gripper/limb/~A" l/r) msg)
     (ros::sleep)
     ))
  (:arm-symbol-to-str
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
      (send coords :transform (send tfl :lookup-transform "base" frame_id (ros::time 0)))))
  (:recognize-bin-boxes-cb
    (msg)
    (let (bin bin-box-pos-z)
      (setq bin-atof-pos-stamp (send (send msg :header) :stamp))
      (setq bin-boxes (make-hash-table))
      (dotimes (i 6)
        (setq bin (elt '(:a :b :c :d :e :f) i))
        (setf (gethash bin bin-boxes) (elt (send msg :boxes) i))
        )
      ;; pos of bin G-L is minus z of A-F
      (dotimes (i 6)
        (setq bin (elt '(:j :k :l :g :h :i) i))
        (setq bin-box (copy-object (elt (send msg :boxes) i)))
        (setq bin-box-pos-z (send (send (send bin-box :pose) :position) :z))
        (send (send (send bin-box :pose) :position) :z (- bin-box-pos-z))
        (sethash bin bin-boxes bin-box)
        )
    ))
  (:recognize-bin-boxes
    ()
    (let ((now (ros::time-now)))
      (ros::subscribe "in_bin_atof/bin_clipper/output/box_array"
                      jsk_recognition_msgs::BoundingBoxArray
                      #'send self :recognize-bin-boxes-cb)
      (while
        (or (null bin-atof-pos-stamp)
            (< (send bin-atof-pos-stamp :sec) (send now :sec)))
        (ros::ros-info "[:recognize-bin-boxes] waiting to synchronize")
        (unix::sleep 1)
        (ros::spin-once))
      (ros::unsubscribe "in_bin_atof/bin_clipper/output/box_array")
      ))
  (:recognize-objects-in-bin-cb
    (msg)
    (setq objects-in-bin-stamp (send (send msg :header) :stamp))
    (sethash objects-in-bin-target objects-in-bin-boxes (send msg :boxes))
    )
  (:recognize-objects-in-bin
    (bin)
    (let ((now (ros::time-now)))
      (ros::subscribe (format nil "in_bin_each_object/bin_~a_cpi_decomposer/boxes" (key-to-str bin))
                      jsk_recognition_msgs::BoundingBoxArray
                      #'send self :recognize-objects-in-bin-cb)
      (setq objects-in-bin-target bin)
      (while
        (or (null objects-in-bin-stamp)
            (< (send objects-in-bin-stamp :sec) (send now :sec)))
        (ros::ros-info "[:recognize-objects-in-bin] Waiting to synchronize")
        (unix::sleep 1)
        (ros::spin-once))
      (ros::unsubscribe (format nil "in_bin_each_object/bin_~a_cpi_decomposer/boxes" (key-to-str bin)))
      ))
  (:ik-avs->object-in-bin
    (arm bin object-index &key (offset #f(0 0 0)))
    (let (object-box coords)
      (setq object-box (elt (gethash bin (send self :get-val 'objects-in-bin-boxes)) object-index))
      (setq coords (send (send self :tf-pose->coords (send object-box :header :frame_id) (send object-box :pose)) :translate offset :world))
      (list (send *baxter* arm :inverse-kinematics coords :rotation-axis nil)
            (send *baxter* arm :inverse-kinematics coords :rotation-axis t))
      ))
  (:send-av
    (&optional (tm 3000))
    (send self :angle-vector (send *baxter* :angle-vector) tm))
  (:ik->bin-entrance
    (arm bin &key (offset #f(0 0 0)))
    (let (bin-box bin-coords bin-dim-x)
      (setq bin-box (gethash bin bin-boxes))
      (unless bin-box
        (ros::ros-error "[:ik->bin-entrance] No data about bin-box ~a. Call :recognize-bin-boxes first." bin)
        (return-from :ik->bin-entrance))
      (setq bin-coords (send self :tf-pose->coords
                             (send (send bin-box :header) :frame_id)
                             (send bin-box :pose)))
      (setq bin-dim-x (m->mm (send (send bin-box :dimensions) :x)))
      (send bin-coords :translate (float-vector (- (/ bin-dim-x 2)) 0 0) :world)
      (send bin-coords :translate offset :world)
      (send *baxter* :fold-pose-lower arm)
      (case bin
        (:a (send *baxter* :fold-pose-upper arm))
        (:c (send *baxter* :fold-pose-upper arm))
        (:j (send *baxter* :fold-pose-middle arm))
        (:k (send *baxter* :fold-pose-middle arm))
        (:l (send *baxter* :fold-pose-middle arm))
        )
      (send *baxter* arm :inverse-kinematics bin-coords :rotation-axis t)))
  (:move-arm-body->bin
    (arm bin)
    (let (avs)
      (pushback (send *baxter* :fold-to-keep-object arm) avs)
      (pushback (send *baxter* :avoid-shelf-pose arm bin) avs)
      (pushback (send self :ik->bin-entrance arm bin :offset #f(-150 0 0)) avs)
      (send self :angle-vector-sequence avs)
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
  (:get-work-order-cb
    (msg)
    (let (one-order)
      (dolist (data (send msg :array))
        (pushback
          (list (str-to-key (send data :bin)) (send data :object))
          one-order))
      (sethash get-work-order-arm work-order one-order)))
  (:get-work-order
    (arm)
    (setq get-work-order-arm arm)
    (ros::subscribe "work_order/left_process"
                    jsk_2015_05_baxter_apc::WorkOrderArray
                    #'send self :get-work-order-cb)
    (unix:sleep 1)
    (ros::spin-once)
    (ros::unsubscribe "work_order/left_process")
    (gethash arm work-order))
  (:get-bin-contents-cb (msg) (setq bin-contents (send msg :array)))
  (:get-bin-contents
    (bin)
    (ros::subscribe "bin_contents"
                    jsk_2015_05_baxter_apc::BinContentsArray
                    #'bin-contents-cb)
    (unix::sleep 1)
    (ros::spin-once)
    (dolist (content bin-contents)
      (when (eq bin (str-to-key (send content :bin)))
        (ros::unsubscribe "bin_contents")
        (return-from :get-bin-contents (send content :objects))))
    )
  )

(defun jsk_2015_05_baxter_apc::baxter-init (&key (ctype :default-controller))
  (unless (boundp '*ri*)
    (setq *ri* (instance jsk_2015_05_baxter_apc::baxter-interface :init :type ctype)))
  (unless (boundp '*baxter*)
    (setq *baxter* (instance jsk_2015_05_baxter_apc::baxter-robot :init)))
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *ri* :calib-grasp :arms)
  )
