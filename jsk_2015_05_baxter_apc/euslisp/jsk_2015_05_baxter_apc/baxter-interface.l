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
  :slots (tfl bin-atof-pos-stamp bin-boxes))

(defmethod jsk_2015_05_baxter_apc::baxter-interface
  (:init
    (&rest args)
    (send-super* :init args)
    (setq tfl (instance ros::transform-listener :init))
    )
  (:start-grasp
   (&optional (arm :arms))
   (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm-to-str arm))))
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
   (dolist (l/r (if (eq arm :arms) (list "left" "right") (list (arm-to-str arm))))
     (ros::advertise-if-yet (format nil "/vacuum_gripper/limb/~A" l/r) std_msgs::Bool 1)
     (setq msg (instance std_msgs::Bool :init))
     (send msg :data nil)
     (ros::ros-info (format nil "stop ~A vacuum gripper" l/r))
     (ros::publish (format nil "/vacuum_gripper/limb/~A" l/r) msg)
     (ros::sleep)
     ))
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
  (:recognize-bin-box-cb
    (msg)
    (let (bin)
      (setq bin-atof-pos-stamp (send (send msg :header) :stamp))
      (setq bin-boxes (make-hash-table))
      (dotimes (i 6)
        (setq bin (elt '(:a :b :c :d :e :f) i))
        (setf (gethash bin bin-boxes) (elt (send msg :boxes) i))
        )
    ))
  (:recognize-bin-box
    (bin)
    (let ((now (ros::time-now)))
      (ros::subscribe "in_bin_atof/bin_clipper/output/box_array"
                      jsk_recognition_msgs::BoundingBoxArray
                      #'send self :recognize-bin-box-cb)
      (while
        (or (null bin-atof-pos-stamp)
            (< (send bin-atof-pos-stamp :sec) (send now :sec)))
        (ros::ros-info "[:recognize-bin-box] waiting to synchronize")
        (unix::sleep 1)
        (ros::spin-once))
      (ros::unsubscribe "in_bin_atof/bin_clipper/output/box_array")
    (gethash bin bin-boxes)))
  (:send-av
    (&optional (tm 3000))
    (send self :angle-vector (send *baxter* :angle-vector) tm))
  (:move-to-bin
    (arm bin)
    (let (bin-box bin-coords bin-dim-x)
      (setq bin-box (send self :recognize-bin-box bin))
      (setq bin-coords (send self :tf-pose->coords
                             (send (send bin-box :header) :frame_id)
                             (send bin-box :pose)))
      (setq bin-dim-x (m->mm (send (send bin-box :dimensions) :x)))
      (send bin-coords :translate (float-vector (- (/ bin-dim-x 2)) 0 0))
      (send *baxter* arm :inverse-kinematics bin-coords :rotation-axis nil)
      (send self :send-av 3000)
      ))
  )

(defun jsk_2015_05_baxter_apc::baxter-init (&key ctype)
  (unless (boundp '*ri*)
    (setq *ri* (instance jsk_2015_05_baxter_apc::baxter-interface :init :type ctype)))
  (unless (boundp '*baxter*)
    (setq *baxter* (instance jsk_2015_05_baxter_apc::baxter-robot :init)))
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *ri* :calib-grasp :arms)
  )
