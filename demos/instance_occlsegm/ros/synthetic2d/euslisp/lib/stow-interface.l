;; vim: set ft=lisp:

(require "package://jsk_arc2017_baxter/euslisp/lib/stow-interface.l")

(unless (find-package "SYNTHETIC2D")
  (make-package "SYNTHETIC2D"))

;; same as jsk_arc2017_baxter::stow-interface other than :recognize-target-object with placing box topic
(defclass synthetic2d::stow-interface
  :super jsk_arc2017_baxter::stow-interface
  :slots (object-pboxes-)
  )
(defmethod synthetic2d::stow-interface
  (:init (&key (moveit nil))
    (send-super :init :moveit moveit)
    (setq object-pboxes- (make-hash-table))
    )
  (:recognize-target-object (arm &key (stamp (ros::time-now)) (timeout 10))
    (let ((box-topic (format nil "~a_hand_camera/cluster_indices_decomposer_target/boxes" (arm2str arm)))
          box-msg
          (pbox-topic (format nil "~a_hand_camera/place/cluster_indices_decomposer_target/boxes" (arm2str arm)))
          pbox-msg
          (com-topic (format nil "~a_hand_camera/cluster_indices_decomposer_target/centroid_pose_array"
                        (arm2str arm)))
          com-msg obj-box obj-pbox obj-coords is-recognized)
      (when (ros::get-param "~data_collection" nil)
        (ros::ros-info-green "Requesting data collection for shelf-bin/tote.")
        (let ((res (ros::service-call (format nil "~a_hand_camera/data_collection_server/save_request" (arm2str arm))
                           (instance std_srvs::TriggerRequest :init))))
          (if (send res :success)
            (ros::ros-info-green "~a" (send res :message))
            (ros::ros-info-red "~a" (send res :message))
            )))
      (ros::subscribe box-topic jsk_recognition_msgs::BoundingBoxArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq box-msg msg)))))
      (ros::subscribe pbox-topic jsk_recognition_msgs::BoundingBoxArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq pbox-msg msg)))))
      (ros::subscribe com-topic geometry_msgs::PoseArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq com-msg msg)))))
      (while (and (not (and box-msg pbox-msg com-msg))
                  (> (+ (send stamp :to-sec) timeout) (send (ros::time-now) :to-sec)))
             (unix::usleep (* 50 1000))
             (ros::spin-once))
      (ros::unsubscribe box-topic)
      (ros::unsubscribe pbox-topic)
      (ros::unsubscribe com-topic)
      (cond
        ((and box-msg pbox-msg com-msg)
         (ros::ros-info "[:recognize-target-object] arm: ~a get cpi msg" arm)
         (setq obj-pbox (send pbox-msg :boxes))
         (setq obj-box (send box-msg :boxes))
         (setq obj-coords
               (mapcar #'(lambda (obj-pose)
                           (send *ri* :tf-pose->coords
                                 (send com-msg :header :frame_id) obj-pose))
                       (send com-msg :poses)))
         (if (and (> (length obj-box) 0) (> (length obj-coords) 0))
           (progn
             (sethash arm object-boxes- obj-box)
             (sethash arm object-pboxes- obj-pbox)
             (sethash arm object-coords- obj-coords)
             (setq is-recognized (send box-msg :header :stamp)))
           (progn
             (ros::ros-error "[:recognize-target-object] arm: ~a obj-box length ~a" arm (length obj-box))
             (ros::ros-error "[:recognize-target-object] arm: ~a obj-coords length ~a" arm (length obj-coords))
             (setq is-recognized nil))))
        (t
         (ros::ros-error "[:recognize-target-object] arm: ~a failed to get cpi msg" arm)
         (setq is-recognized nil)))
      is-recognized))
  )

(defun synthetic2d::stow-init (&key (ctype :default-controller) (moveit nil))
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
      (setq *ti* (instance synthetic2d::stow-interface :init :moveit moveit)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)))
