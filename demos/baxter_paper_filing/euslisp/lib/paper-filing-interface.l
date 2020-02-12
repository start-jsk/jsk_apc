;; -*- mode: lisp;-*-

(require "package://sphand_driver/euslisp/lib/baxterlgv8-interface.l")


(ros::load-ros-package "geometry_msgs")
(ros::load-ros-package "jsk_recognition_msgs")

(unless (find-package "BAXTER_PAPER_FILING")
  (make-package "BAXTER_PAPER_FILING"))

(defclass baxter_paper_filing::paper-filing-interface
  :super propertied-object
  :slots (moveit-p-
          table-cube-
          gripper-dims-
          paper-boxes-
          paper-coords-
          paper-thre-l-
          paper-thre-r-
          paper-refl-min-l-
          paper-refl-min-r-
          paper-refl-max-l-
          paper-refl-max-r-
          ))

(defmethod baxter_paper_filing::paper-filing-interface
  (:init (&key (moveit nil))
    (setq moveit-p- moveit)
    (setq table-cube- nil)
    ;; dimensions of grippers
    (setq gripper-dims- (make-hash-table))
    (sethash :pad-r gripper-dims- (make-hash-table))
    (sethash :larm (gethash :pad-r gripper-dims-) 15)
    (sethash :rarm (gethash :pad-r gripper-dims-) 15)
    (setq paper-boxes- nil)
    (setq paper-coords- nil)
    (setq paper-thre-l- 25 paper-thre-r- 25)
    (setq paper-refl-min-l- 10000.0 paper-refl-min-r- 10000.0)
    (setq paper-refl-max-l- 2000000.0 paper-refl-max-r- 2000000.0))
  (:bbox->cube (bbox)
    (let* ((dims (ros::tf-point->pos (send bbox :dimensions)))
           (bx (make-cube (elt dims 0) (elt dims 1) (elt dims 2))))
      (send bx :newcoords
            (send *ri* :tf-pose->coords
                  (send bbox :header :frame_id)
                  (send bbox :pose)))
      bx))
  (:recognize-table-box
    (&key (stamp (ros::time-now)))
    (ros::ros-info "[:recognize-table-box] Recognizing table box...")
    (let ((box-topic (format nil "publish_table_boxes/output"))
          box-msg box-list bin-list)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[:recognize-table-box] Recognized table box")
          (setq table-cube- (send self :bbox->cube (car (send box-msg :boxes)))))
        (ros::ros-fatal "[:recognize-table-box] Cannot recognize table box"))))
  (:ik->cube-center
    (arm cube &key (offset #f(0 0 0)) (rpy #f(0 0 0)) (rotation-axis t)
         (use-gripper nil) (move-palm-end nil) (use-rotate-wrist-ik nil) (parallel-with :local))
    (let (cube-coords)
      (setq cube-coords (send cube :copy-worldcoords))
      (send cube-coords :translate
            (float-vector 0.0 0.0 (/ (z-of-cube cube) 2.0))
            :local)
      (send cube-coords :translate offset :local)
      (send cube-coords :rotate (aref rpy 0) :z)
      (send cube-coords :rotate (aref rpy 1) :y)
      (send cube-coords :rotate (aref rpy 2) :x)
      (when (eq parallel-with :world)
        (setq cube-coords (make-coords :pos (send cube-coords :worldpos))))
      (if use-rotate-wrist-ik
        (send *baxter* :rotate-wrist-ik arm cube-coords
              :rotation-axis rotation-axis
              :use-gripper use-gripper
              :move-palm-end move-palm-end)
        (send *baxter* arm :inverse-kinematics cube-coords
              :rotation-axis rotation-axis
              :use-gripper use-gripper
              :move-palm-end move-palm-end))))
  (:move-larm->recognition-pose ()
    (ros::ros-info "[:move-larm->recognition-pose] Moving larm to recognition pose")
    (send *baxter* :reset-pose :larm)
    (send *baxter* :rotate-gripper :larm -90 :relative nil)
    (send self :ik->cube-center :larm table-cube-
          :offset #f(30 290 250) :rpy (float-vector 0 -pi/2 pi/2)
          :use-gripper nil :use-rotate-wrist-ik t)
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000
          (send *ri* :get-arm-controller :larm) 0))
  (:recognize-paper (&key (stamp (ros::time-now)) (timeout *inf*))
    (ros::ros-info "[:recognize-paper] Recognizing paper on table...")
    (let ((box-topic "left_hand_camera/cluster_indices_decomposer_paper/boxes")
          box-msg
          (com-topic "left_hand_camera/cluster_indices_decomposer_paper/centroid_pose_array")
          com-msg recognized-p)
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
      (while (and (not (setq recognized-p (and box-msg com-msg
                                               (> (length (send box-msg :boxes)) 0)
                                               (> (length (send com-msg :poses)) 0))))
                  (> (+ (send stamp :to-sec) timeout) (send (ros::time-now) :to-sec)))
        (unix::usleep (* 50 1000))
        (ros::spin-once))
      (when recognized-p
        (ros::ros-info "[:recognize-paper] Get first result, but waiting for stable result")
        ;; Get stable result
        (setq stamp (ros::time+ (ros::time-now) (ros::time 3)))
        (setq box-msg nil com-msg nil)
        (while (and (not (setq recognized-p (and box-msg com-msg
                                                 (> (length (send box-msg :boxes)) 0)
                                                 (> (length (send com-msg :poses)) 0))))
                    (> (+ (send stamp :to-sec) timeout) (send (ros::time-now) :to-sec)))
          (unix::usleep (* 50 1000))
          (ros::spin-once)))
      (ros::unsubscribe box-topic)
      (ros::unsubscribe com-topic)
      (if recognized-p
        (progn
          (ros::ros-info "[:recognize-paper] Get cpi msg including bbox and com")
          (setq paper-boxes- (send box-msg :boxes))
          (setq paper-coords-
                (mapcar #'(lambda (obj-pose)
                            (send *ri* :tf-pose->coords
                                  (send com-msg :header :frame_id) obj-pose))
                        (send com-msg :poses)))
          t)
        (progn
          (ros::ros-error "[:recognize-paper] Failed to get cpi msg including bbox and com")
          nil))))
  (:slide-paper ()
    (ros::ros-info "[:slide-paper] Sliding paper to grasp...")
    ;; Turn off unused proximities. Don't forget to turn on afterwords
    (send *ri* :turn-off-proximity :intensity :larm
          (list :left-root :right-root :palm-right :palm-left))
    (send *ri* :turn-off-proximity :tof :larm
          (list :left-root :right-root :palm-right :palm-left))
    (unix::sleep 1)
    (send *ri* :calib-proximity-threshold :larm)
    (send *ri* :reset-i-prox-calib :larm)
    (send *ri* :move-hand :larm
          (send *baxter* :hand-grasp-pre-pose :larm :cylindrical) 1000 :wait nil)
    (let ((paper-cube (send self :bbox->cube (car paper-boxes-)))
          contact-coords target-protrusion offset-from-edge above-coords slide-l gripper-l avs)
      (setq contact-coords (send paper-cube :copy-worldcoords))
      ;; Decide y position from paper edge
      (setq target-protrusion 50)
      (setq offset-from-edge (+ target-protrusion
                                (gethash :larm (gethash :pad-r gripper-dims-))))
      (send contact-coords :translate
            (float-vector 0 (- (/ (y-of-cube paper-cube) 2) offset-from-edge) 0)
            :local)
      ;; Decide z position from table-cube- as recognition result may be unstable
      (setf (elt (send contact-coords :worldpos) 2) (+ (elt (send table-cube- :worldpos) 2)
                                                       (/ (z-of-cube table-cube-) 2) -20))
      ;; Rotate to match end-coords
      (send contact-coords :rotate (deg2rad -90) :z)
      (setq above-coords (copy-object contact-coords))
      (send above-coords :translate #f(0 0 150))
      (setq slide-l
            (- (+ (elt (send table-cube- :worldpos) 1) (/ (y-of-cube table-cube-) 2))
               (elt (send contact-coords :worldpos) 1) (gethash :larm (gethash :pad-r gripper-dims-))
               30))
      (ros::ros-info "[:slide-paper] above-coords: ~a -> contact-coords: ~a -> slide-l: ~a"
                     above-coords contact-coords slide-l)
      ;; Solve IK
      (send *baxter* :rotate-gripper :larm 0 :relative nil)
      (send *baxter* :slide-gripper :larm 0 :relative nil)
      ;; Move gripper above paper
      (send *ri* :angle-vector-raw (send *baxter* :rotate-wrist-ik :larm above-coords)
            :fast (send *ri* :get-arm-controller :larm) 0 :scale 3.0)
      (send *ri* :wait-interpolation)
      (send *baxter* :slide-gripper :larm (setq gripper-l 30) :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :larm :inverse-kinematics contact-coords)
            :fast (send *ri* :get-arm-controller :larm) 0 :scale 6.0)
      (send *ri* :wait-interpolation)
      ;; Slide paper
      (send *ri* :angle-vector-raw
            (send *baxter* :larm :move-end-pos (float-vector (- slide-l) 0 0) :local)
            :fast (send *ri* :get-arm-controller :larm) 0 :scale 30.0)
      (send *ri* :wait-interpolation)
      ;; Move gripper to grasp position
      (send *ri* :angle-vector-raw (send *baxter* :slide-gripper :larm 0 :relative nil)
            1000 (send *ri* :get-arm-controller :larm))
      (send *ri* :wait-interpolation)
      (setq avs
            (list (send *baxter* :larm :move-end-pos
                        (float-vector (- 10 target-protrusion) 0 0) :local)
                  (send *baxter* :larm :move-end-pos (float-vector 0 0 (- 3 gripper-l)) :local)))
      (send *ri* :angle-vector-sequence-raw avs
            :fast (send *ri* :get-arm-controller :larm) 0 :scale 20.0)
      (send *ri* :wait-interpolation)))
  (:grasp-paper ()
    (ros::ros-info "[:grasp-paper] Grasping paper...")
    (let ((roll-jt-b (aref (send *ri* :get-real-finger-av :larm) 2))
          tm-to-grasp l-dist r-dist l-refl r-refl)
      (when (< roll-jt-b 110.0)
        (ros::ros-info "[:grasp-paper] roll-jt-b is ~a < 110, so initialize finger" roll-jt-b)
        ;; Turn off flex reflex
        (send *ri* :enable-flex-reflex :larm nil)
        (send *ri* :reset-i-prox-calib :larm)
        (send *ri* :move-hand :larm
              (send *baxter* :hand :larm :angle-vector #f(77 90 110)) 2000)
        (setq roll-jt-b 110.0))
      (setq tm-to-grasp (* 3000 (/ (- 180.0 roll-jt-b) 70)))  ;; Default value is 3000
      (ros::ros-info "[:grasp-paper] tm-to-grasp: ~a" tm-to-grasp)
      ;; Close finger until sensing paper
      ;; 160 is too far and 2000 is too fast to cancel.
      ;; After wait-interpolation, sensor values drop under threshold.
      ;; (send *ri* :move-hand :larm
      ;;       (send *baxter* :hand :larm :angle-vector #f(77 90 160)) 2000 :wait nil)
      (send *ri* :move-hand :larm
            (send *baxter* :hand :larm :angle-vector #f(77 90 180)) tm-to-grasp :wait nil)
      (send *ri* :wait-interpolation-until-funcs :larm
            (list #'(lambda ()
                      (setq l-refl (send *ri* :get-i-reflectance-param :larm :left-tip))
                      (setq r-refl (send *ri* :get-i-reflectance-param :larm :right-tip))
                      (and
                        (< (setq l-dist (send *ri* :get-i-prox-distance :larm :left-tip))
                           paper-thre-l-)
                        (< (setq r-dist (send *ri* :get-i-prox-distance :larm :right-tip))
                           paper-thre-r-))))
            :hand)
      (ros::ros-info "[:grasp-paper] Final dist: l: ~a r: ~a" l-dist r-dist)
      (ros::ros-info "[:grasp-paper] Final refl: l: ~a r: ~a" l-refl r-refl)
      (if (or (< l-refl paper-refl-min-l-) (< r-refl paper-refl-min-r-)
              (> l-refl paper-refl-max-l-) (> r-refl paper-refl-max-r-))
        (progn
          (ros::ros-error "[:grasp-paper] Nothing is grasped")
          nil)
        (progn
          (ros::ros-info "[:grasp-paper] Paper is grasped")
          (if (and (< l-dist paper-thre-l-) (< r-dist paper-thre-r-))
            t
            (progn
              (ros::ros-error "[:grasp-paper] However, fingers did not stop")
              nil))))))
  (:return-from-grasp-paper ()
    (ros::ros-info "[:return-from-grasp-paper] Returning from :grasp-paper...")
    (send *ri* :move-hand :larm
          (send *baxter* :hand-grasp-pre-pose :larm :cylindrical) 1000 :wait t)
    (send *ri* :angle-vector-raw
          (send *baxter* :larm :move-end-pos #f(0 0 200) :local)
          :fast (send *ri* :get-arm-controller :larm) 0 :scale 6.0)
    (send *ri* :wait-interpolation)
    ;; Turn on flex reflex
    (send *ri* :enable-flex-reflex :larm t)
    ;; Turn on all proximities
    (send *ri* :turn-off-proximity :intensity :larm nil)
    (send *ri* :turn-off-proximity :tof :larm nil))
  (:file-paper ()
    (ros::ros-info "[:file-paper] Filing paper...")
    (send *ri* :angle-vector-raw
          (send *baxter* :larm :move-end-pos #f(0 0 150) :local)
          :fast (send *ri* :get-arm-controller :larm) 0 :scale 6.0)
    (send *ri* :wait-interpolation)
    (assert (send *baxter* :larm :move-end-pos #f(450 0 0) :local))
    (let (l-dist r-dist (thre-l (+ paper-thre-l- 10)) (thre-r (+ paper-thre-r- 10)))
      (while
        (progn
          (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
                :fast (send *ri* :get-arm-controller :larm) 0 :scale 7.0)
          (ros::ros-info "[:file-paper] Now you can pull out paper")
          (send *ri* :wait-interpolation-until-funcs :larm
                (list #'(lambda ()
                          (or
                            (>= (setq l-dist (send *ri* :get-i-prox-distance :larm :left-tip))
                                thre-l)
                            (>= (setq r-dist (send *ri* :get-i-prox-distance :larm :right-tip))
                                thre-r)))))
          (if (or (>= l-dist thre-l) (>= r-dist thre-r))
            (progn
              (ros::ros-error "[:file-paper] Paper is pulled out")
              (while
                (progn
                  ;; Slightly open finger
                  (send *ri* :move-hand :larm
                        (send *baxter* :hand :larm :angle-vector #f(77 90 140)) 1000)
                  (ros::ros-info "[:file-paper] Waiting for paper to be inserted...")
                  (while (or (>= (send *ri* :get-i-prox-distance :larm :left-tip) 70)
                             (>= (send *ri* :get-i-prox-distance :larm :right-tip) 70))
                         (unix::usleep (* 50 1000)))
                  (ros::ros-info "[:file-paper] Paper is inserted")
                  (not (send self :grasp-paper))))
              t)
            nil))))
    (ros::ros-info "[:file-paper] Now you cannot pull out paper")
    (let (avs)
      (setq avs (list (send *baxter* :larm :move-end-pos #f(0 0 -95) :local)
                      (send *baxter* :larm :move-end-pos #f(50 0 0) :local)))
      (send *ri* :angle-vector-sequence-raw avs
            :fast (send *ri* :get-arm-controller :larm) 0 :scale 20.0)
      (send *ri* :wait-interpolation)))
  (:return-from-file-paper ()
    (ros::ros-info "[:return-from-file-paper] Returning from :file-paper...")
    (send *ri* :move-hand :larm
          (send *baxter* :hand :larm :angle-vector #f(77 90 110)) 1000 :wait nil)
    (send *ri* :angle-vector-raw (send *baxter* :larm :move-end-rot 10 :y :local)
          :fast (send *ri* :get-arm-controller :larm) 0 :scale 4.0)
    (send *ri* :wait-interpolation)
    (send *ri* :move-hand :larm
          (send *baxter* :hand-grasp-pre-pose :larm :cylindrical) 1000)
    (send *baxter* :larm :move-end-rot -10 :y :local)
    (send *ri* :angle-vector-raw
          (send *baxter* :larm :move-end-pos #f(0 0 200) :local)
          :fast (send *ri* :get-arm-controller :larm) 0 :scale 10.0)
    (send *ri* :wait-interpolation)
    ;; Turn on flex reflex
    (send *ri* :enable-flex-reflex :larm t)
    ;; Turn on all proximities
    (send *ri* :turn-off-proximity :intensity :larm nil)
    (send *ri* :turn-off-proximity :tof :larm nil))
  (:mainloop-step ()
    (send *ri* :move-hand :larm
          (send *baxter* :hand-grasp-pre-pose :larm :opposed) 1000 :wait nil)
    (send self :move-larm->recognition-pose)
    (send *ri* :wait-interpolation)
    (send self :recognize-paper)
    (send self :slide-paper)
    (if (send self :grasp-paper)
      (progn
        (send self :file-paper)
        (send self :return-from-file-paper)
        t)
      (progn
        (send self :return-from-grasp-paper)
        nil)))
  (:mainloop ()
    (send self :recognize-table-box)
    (while (ros::ok)
      (send self :mainloop-step)
      (ros::spin-once))))

(defun baxter_paper_filing::paper-filing-init (&key (ctype :default-controller) (moveit nil))
  (let (mvit-env mvit-rb)
    (when moveit
      (setq mvit-env (instance sphand_driver::baxterlgv8-moveit-environment))
      (setq mvit-rb (instance jsk_arc2017_baxter::baxterlgv7-robot :init)))
    (unless (boundp '*ri*)
      (setq *ri* (instance sphand_driver::baxterlgv8-interface :init :type ctype
                           :moveit-environment mvit-env
                           :moveit-robot mvit-rb)))
    (unless (boundp '*baxter*)
      (setq *baxter* (instance jsk_arc2017_baxter::baxterlgv7-robot :init)))
    (unless (boundp '*co*)
      (setq *co* (when moveit (instance collision-object-publisher :init))))
    (unless (boundp '*ti*)
      (setq *ti* (instance baxter_paper_filing::paper-filing-interface :init :moveit moveit)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))))
