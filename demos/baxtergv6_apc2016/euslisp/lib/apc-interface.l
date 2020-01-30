;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")
(require "package://baxtergv6_apc2016/euslisp/lib/baxter-interface.l")


(ros::load-ros-package "geometry_msgs")
(ros::load-ros-package "jsk_arc2017_baxter")
(ros::load-ros-package "jsk_arc2017_common")
(ros::load-ros-package "jsk_gui_msgs")
(ros::load-ros-package "jsk_recognition_msgs")
(ros::load-ros-package "std_srvs")

(unless (find-package "BAXTERGV6_APC2016")
  (make-package "BAXTERGV6_APC2016"))

(defclass baxtergv6_apc2016::apc-interface
  :super propertied-object
  :slots (attached-objects-
          bbox-volume-threshold-
          bin-cubes-
          bin-list-
          bin-movable-regions-
          bin-object-volumes-
          cardboard-cubes-
          cardboard-list-
          finished-objects-
          object-boxes-
          object-coords-
          orders-
          postponed-objects-
          rack-leg-cubes-
          tote-cubes-
          tote-movable-regions-
          unfinished-objects-
          gripper-dims-
          ))

(defmethod baxtergv6_apc2016::apc-interface
  (:init ()
    ;; initialize slots
    (setq bbox-volume-threshold- 0.01) ;; [m^3]
    (setq attached-objects- (make-hash-table))
    (setq bin-cubes- (make-hash-table))
    (setq bin-movable-regions- (make-hash-table))
    (setq bin-object-volumes- (make-hash-table))
    (setq object-boxes- (make-hash-table))
    (setq object-coords- (make-hash-table))
    (setq cardboard-cubes- (make-hash-table))
    (setq tote-cubes- (make-hash-table))
    (setq tote-movable-regions- (make-hash-table))
    (setq postponed-objects- (make-hash-table))
    (setq rack-leg-cubes- (make-hash-table))
    (setq finished-objects- (make-hash-table))
    (setq unfinished-objects- (make-hash-table))
    ;; dimensions of grippers
    ;; TODO: decide values
    (setq gripper-dims- (make-hash-table))
    (sethash :pad-r gripper-dims- (make-hash-table))
    (sethash :larm (gethash :pad-r gripper-dims-) 15)
    (sethash :rarm (gethash :pad-r gripper-dims-) 15)
    (sethash :pad-link-l gripper-dims- (make-hash-table))
    (sethash :larm (gethash :pad-link-l gripper-dims-) 45)
    (sethash :rarm (gethash :pad-link-l gripper-dims-) 45)
    (sethash :tube-t gripper-dims- (make-hash-table))
    (sethash :larm (gethash :tube-t gripper-dims-) 28)
    (sethash :rarm (gethash :tube-t gripper-dims-) 28)
    (sethash :cylindrical-max-t gripper-dims- (make-hash-table))
    (sethash :larm (gethash :cylindrical-max-t gripper-dims-) 90)
    (sethash :rarm (gethash :cylindrical-max-t gripper-dims-) 90)
    (sethash :opposed-max-w gripper-dims- (make-hash-table))
    (sethash :larm (gethash :opposed-max-w gripper-dims-) 160)
    (sethash :rarm (gethash :opposed-max-w gripper-dims-) 160)
    (sethash :pad-to-finger gripper-dims- (make-hash-table))
    (sethash :larm (gethash :pad-to-finger gripper-dims-) 60)
    (sethash :rarm (gethash :pad-to-finger gripper-dims-) 60))
  (:arc-reset-pose
    (&optional (arm :arms) &rest args)
    (dolist (tmp-arm (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (send *baxter* :arc-reset-pose tmp-arm))
    (send* *ri* :angle-vector (send *baxter* :angle-vector) args))
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
    ;; movable-region: ((min-of-x max-of-x) (min-of-y max-of-y) (min-of-z max-of-z) center-coords)
    ;; rule of coords: z is direction from center to entrance
    (let (movable-region)
      (pushback (list
                  (+ (/ (x-of-cube cube) -2) (elt offset 0))
                  (- (/ (x-of-cube cube) 2) (elt offset 0)))
                movable-region)
      (pushback (list
                  (+ (/ (y-of-cube cube) -2) (elt offset 1))
                  (- (/ (y-of-cube cube) 2) (elt offset 1)))
                movable-region)
      (pushback (list
                  (+ (/ (z-of-cube cube) -2) (elt offset 2))
                  (- (/ (z-of-cube cube) 2) (elt offset 2)))
                movable-region)
      (pushback (send cube :copy-worldcoords) movable-region)
      movable-region))
  (:set-movable-region-for-bin
    (&key (offset (list 0 0 0)))
    (let (cube)
        (progn
          (dolist (bin bin-list-)
               (setq cube (gethash bin bin-cubes-))
               (sethash bin bin-movable-regions-
                        (send self :cube->movable-region cube :offset offset))))))
  (:set-movable-region-for-tote
    (arm &key (offset (list 0 0 0)))
    (let (cube)
      (setq cube (gethash arm tote-cubes-))
      (sethash arm tote-movable-regions- (send self :cube->movable-region cube :offset offset))))
  (:get-next-target-bin (arm &key (stamp (ros::time-now)))
    (when (eq (length orders-) 0)
      (ros::ros-error "[:get-next-target-bin] no work-order, try :get-work-orders first")
      (return-from :get-next-target-bin nil))
    (let (bin-contents target-bins postponed-bins msg)
      (setq msg (one-shot-subscribe "/json_saver/output/bin_contents"
                                    jsk_arc2017_common::ContentArray
                                    :timeout 10000
                                    :after-stamp stamp))
      (setq bin-contents
            (mapcar #'(lambda (content)
                        (cons (str2symbol (send content :bin))
                              (send content :items)))
                    (send msg :contents)))
      (ros::ros-debug "[:get-next-target-bin] get bin-contents ~A" bin-contents)
      (setq target-bins
            (remove-duplicates
              (remove nil
                      (mapcar
                        #'(lambda (order)
                            (let ((item (send order :item))
                                  (bin (str2symbol (send order :bin))))
                              (ros::ros-debug "[:get-next-target-bin] item ~A, bin ~A" item bin)
                              (when (find item (cdr (assoc bin bin-contents)) :test #'string=)
                                (unless
                                  (or (find item (gethash bin finished-objects-) :test #'string=)
                                      (find item (gethash bin postponed-objects-) :test #'string=))
                                  bin))))
                        orders-))))
      ;; 66% random target
      ;; 33% select unfinished-objects (never tried to recognize)
      (when (< (random 3) 2) (setq target-bins nil))
      (ros::ros-debug "[:get-next-target-bin] get target-bins ~A" target-bins)
      (when target-bins
        (return-from :get-next-target-bin
                     (elt target-bins (random (length target-bins)))))
      (setq postponed-bins
            (remove-duplicates
              (remove nil
                      (mapcar
                        #'(lambda (order)
                            (let ((item (send order :item))
                                  (bin (str2symbol (send order :bin))))
                              (ros::ros-debug "[:get-next-target-bin]   item ~A, bin ~A" item bin)
                              (when (find item (cdr (assoc bin bin-contents)) :test #'string=)
                                (unless
                                  (find item (gethash bin finished-objects-) :test #'string=)
                                  bin))))
                        orders-))))

      (ros::ros-debug "[:get-next-target-bin] get postponed bins ~A" postponed-bins)
      (when postponed-bins
        (return-from :get-next-target-bin
                     (elt postponed-bins (random (length postponed-bins)))))
      nil))
  (:get-work-orders
    (arm)
    (let (msg)
      (setq msg
        (one-shot-subscribe
          (format nil "/strategic_work_order/~a_hand" (arm2str arm))
          jsk_arc2017_common::WorkOrderArray))
      (setq orders- (send msg :orders))))
  (:get-next-work-order (arm bin)
    (when (eq (length orders-) 0)
      (ros::ros-error "[:get-next-work-order] There is no order")
      (return-from :get-next-work-order nil))
    ;; return never tried object
    (dolist (skipping-objects
              (list (append (gethash bin finished-objects-) (gethash bin postponed-objects-))
                    (gethash bin finished-objects-)))
      (dolist (order orders-)
        (unless (find (send order :item) skipping-objects :test #'string=)
          (return-from :get-next-work-order order))))
    nil)
  (:get-certain-work-order
    (arm bin)
    (when (eq (length orders-) 0)
      (ros::ros-error "[:get-certain-work-order] There is no order")
      (return-from :get-certain-work-order nil))
    (when (null bin) (return-from :get-certain-work-order (elt orders- 0)))
    (dotimes (i (length orders-))
      (when (string= (send (elt orders- i) :bin)
                     (symbol-string bin))
        (return-from :get-certain-work-order (elt orders- i)))))
  (:select-work-order-in-bin
    (arm bin &key (stamp (ros::time-now)) (timeout 10))
    (let ((box-topic (format nil "~a_hand_camera/cluster_indices_decomposer_label/boxes" (arm2str arm)))
          (candidates-topic (format nil "/~a_hand_camera/candidates_publisher/output/candidates" (arm2str arm)))
          box-msg box-objects box-labels bin-orders
          candidates-msg candidate-labels label-names
          unfinished-orders)
      (setq bin-orders
            (remove nil (mapcar
                          #'(lambda (order)
                              (when (string= (send order :bin) (symbol-string bin)) order))
                          orders-)))
      (setq label-names
            (ros::get-param (format nil "/~a_hand_camera/label_names"
                                    (arm2str arm))))
      (ros::subscribe box-topic jsk_recognition_msgs::BoundingBoxArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq box-msg msg)))))
      (ros::subscribe candidates-topic jsk_recognition_msgs::LabelArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq candidates-msg msg)))))
      (while (and (not (and box-msg candidates-msg))
                  (> (+ (send stamp :to-sec) timeout)
                     (send (ros::time-now) :to-sec)))
             (unix::usleep (* 50 1000))
             (ros::spin-once))
      (ros::unsubscribe box-topic)
      (ros::unsubscribe candidates-topic)

      (when (and box-msg candidates-msg)
        (setq box-labels
              (remove nil (mapcar
                            #'(lambda (box)
                                (let (volume)
                                  (setq volume (* (send box :dimensions :x)
                                                  (send box :dimensions :y)
                                                  (send box :dimensions :z)))
                                  (when (> volume 0.0)
                                    (send box :label))))
                            (send box-msg :boxes))))
        (setq candidate-labels
              (mapcar #'(lambda (label-msg) (send label-msg :id)) (send candidates-msg :labels)))
        (setq box-labels
              (remove nil (mapcar
                            #'(lambda (label)
                                (when (find label candidate-labels) label))
                            box-labels)))
        (unless box-labels (return-from :select-work-order-in-bin nil))
        (setq box-objects (mapcar #'(lambda (label) (elt label-names label)) box-labels))
        (ros::ros-debug "arm: ~A, box-objects: ~A" (arm2str arm) box-objects)
        (dolist (order bin-orders)
          (let ((object-name (send order :item)))
            (when (and (find object-name box-objects :test #'string=)
                       (not (find object-name (append (gethash bin finished-objects-)
                                                      (gethash bin postponed-objects-))
                                  :test #'string=)))
              (return-from :select-work-order-in-bin order))))
        (setq unfinished-orders
              (remove nil
                      (mapcar #'(lambda (order)
                                  (let ((object-name (send order :item)))
                                    (when (and (find object-name box-objects :test #'string=)
                                               (not (find object-name
                                                          (gethash bin finished-objects-)
                                                          :test #'string=)))
                                      order)))
                      bin-orders)))
        (when unfinished-orders
          (return-from :select-work-order-in-bin
                       (elt unfinished-orders (random (length unfinished-orders))))))
      nil))
  (:check-exist-in-order (obj)
    (let (index)
      (setq index
            (position obj
                      (mapcar #'(lambda (order) (send order :item)) orders-) :test #'string=))
      (if index
        (elt orders- index)
        nil)))
  (:check-bin-exist (bin) (if (gethash bin bin-cubes-) t nil))
  (:recognize-bin-boxes
    (&key (stamp (ros::time-now)))
    (let ((box-topic (format nil "transformable_bin_markers/output/boxes"))
          box-msg box-list bin)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[~a] [:recognize-bin-boxes] recognize bin boxes" (ros::get-name))
          (setq box-list (send box-msg :boxes))
          (dotimes (i (length box-list))
            (setq bin (str2symbol (format nil "~c" (+ #\a i))))
            (sethash bin bin-cubes- (send self :bbox->cube (elt box-list i)))
            ;; generate bin-list- dynamically
            (pushback bin bin-list-)
            ;; initialize bin-object-volumes- if not initialized
            (if (null (gethash bin bin-object-volumes-))
              (sethash bin bin-object-volumes- 0))))
        (ros::ros-fatal "[:recognize-bin-boxes] cannot recognize bin boxes"))))
  (:recognize-cardboard-boxes
    (&key (stamp (ros::time-now)))
    (let ((box-topic (format nil "transformable_cardboard_markers/output/boxes"))
          box-msg box-list cardboard)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[~a] [:recognize-cardboard-boxes] recognize cardboard boxes" (ros::get-name))
          (setq box-list (send box-msg :boxes))
          (dotimes (i (length box-list))
            (setq cardboard (str2symbol (format nil "~c" (+ #\a i))))
            (sethash bin bin-cubes- (send self :bbox->cube (elt box-list i)))
            (sethash cardboard cardboard-cubes- (send self :bbox->cube (elt box-list i)))
            ;; generate cardboard-list- dynamically
            (pushback cardboard cardboard-list-)))
        (ros::ros-fatal "[:recognize-cardboard-boxes] cannot recognize cardboard boxes"))))
  (:recognize-tote-boxes
    (&key (stamp (ros::time-now)))
    (let ((box-topic (format nil "transformable_tote_markers/output/boxes"))
                     box-msg box-list)
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (if box-msg
        (progn
          (ros::ros-info "[~a] [:recognize-tote-boxes] recognize tote bbox" (ros::get-name))
          (setq box-list (send box-msg :boxes))
          (dolist (arm '(:larm :rarm))
            (sethash arm tote-cubes- (send self :bbox->cube (pop box-list)))))
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
        ((eq box-type :cardboard) (setq box-cubes cardboard-cubes- box-color :yellow))
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
    (arm bin &key (stamp (ros::time-now)) (n-random nil))
    (let ((box-topic (format nil "~a_hand_camera/cluster_indices_decomposer_label/boxes" (arm2str arm)))
          box-msg box-labels label-names)
      (setq label-names
            (ros::get-param (format nil "/~a_hand_camera/label_names"
                                    (arm2str arm))))
      (setq box-msg (one-shot-subscribe box-topic
                                        jsk_recognition_msgs::BoundingBoxArray
                                        :timeout 10000
                                        :after-stamp stamp))
      (when box-msg
        (setq box-labels
              (remove nil (mapcar
                            #'(lambda (box)
                                (let (volume)
                                  (setq volume (* (send box :dimensions :x)
                                                  (send box :dimensions :y)
                                                  (send box :dimensions :z)))
                                  (when (> volume 0.0)
                                    (send box :label))))
                            (send box-msg :boxes))))
        (unless box-labels (return-from :get-largest-object-index nil))
        (dolist (skipping-objects
                  (list (append (gethash bin finished-objects-) (gethash bin postponed-objects-))
                        (gethash bin finished-objects-)))
          (let (label-indices target-indices target-index)
            (setq label-indices
                  (remove nil (mapcar
                                #'(lambda (label-index)
                                    (let ((obj-name (elt label-names label-index)))
                                      (unless (find obj-name skipping-objects :test #'string=)
                                        label-index)))
                                box-labels)))
            (when label-indices
              (if n-random
                (progn
                  (setq target-indices (subseq label-indices 0 n-random))
                  (setq target-index (elt target-indices (random (length target-indices)))))
                (setq target-index (car label-indices)))
              (return-from :get-largest-object-index target-index)))))
      nil))
  (:resolve-collision-between-fingers (arm)
    (send *ri* :angle-vector-raw
          (send *baxter* :slide-gripper arm 120 :relative nil)
          1000 (send *ri* :get-arm-controller arm) 0)
    (send *ri* :move-hand arm
          (send *baxter* :hand-grasp-pre-pose arm :cylindrical) 1000 :wait nil)
    (send *ri* :wait-interpolation)
    (unix::sleep 1)
    (send *ri* :angle-vector-raw
          (send *baxter* :slide-gripper arm 0 :relative nil)
          1000 (send *ri* :get-arm-controller arm) 0)
    (send *ri* :move-hand arm
          (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000 :wait nil)
    (send *ri* :wait-interpolation))
  (:pick-object-in-bin
    (arm bin &rest args)
    (let (pick-result movable-region)
      (setq movable-region (gethash bin bin-movable-regions-))
      (unless movable-region
        (ros::ros-error "[:pick-object-in-bin] No movable region for bin. Call :set-movable-region-for-bin first.")
        (return-from :pick-object-in-bin nil))
      (setq pick-result (send* self :pick-object-with-movable-region arm movable-region args))
      pick-result))
  (:pick-object-in-tote
    (arm &rest args)
    (let (pick-result movable-region)
      (setq movable-region (gethash arm tote-movable-regions-))
      (unless movable-region
        (ros::ros-error "[:pick-object-in-tote] No movable region for tote. Call :set-movable-region-for-tote first.")
        (return-from :pick-object-in-tote nil))
      (setq pick-result (send* self :pick-object-with-movable-region arm movable-region args))
      pick-result))
  (:pick-object-with-movable-region
    (arm movable-region &key (n-trial 1) (n-trial-same-pos 1)
         (do-stop-grasp nil) (grasp-style :suction))
    "Return value: (list pick-result support-style)
     pick-result: :grasp-succeeded, :grasp-succeeded-with-support,
                  :grasp-failed, :ik-failed or :plan-failed
     support-style: nil, :cylindrical, :opposed or :scrape-out"
    (send *ri* :stop-grasp arm :pinch)
    (send *ri* :calib-proximity-threshold arm)
    (let (pick-result avs object-index obj-local-pos obj-cube obj-local-cube approach-rpy as-table)
      ;; pick largest bounding box
      (setq object-index 0)
      ;; obj-local-pos: relative position to movable-region coords
      (setq obj-local-pos
            (send self :get-object-position arm movable-region :object-index object-index))
      ;; get object cube in worldcoords
      (setq obj-cube (send self :bbox->cube (elt (gethash arm object-boxes-) object-index)))
      (sethash arm attached-objects- (copy-object obj-cube))
      ;; convert object cube from world to local
      (setq obj-local-cube (copy-object obj-cube))
      (send obj-local-cube :move-to
            (send (elt movable-region 3) :transformation
                  (send obj-local-cube :worldcoords) :local) :world)
      (ros::ros-info "obj-local-pos: ~a, obj-local-cube: ~a, movable-region: ~a, grasp-style: ~a"
                     obj-local-pos obj-local-cube movable-region grasp-style)
      (setq as-table
            (send self :decide-approach-and-support
                  arm obj-local-pos obj-local-cube movable-region grasp-style))
      (unless as-table
            (ros::ros-error "[:pick-object-with-movable-region] arm:~a Abort picking" arm)
            (return-from :pick-object-with-movable-region (list :plan-failed nil)))
      (ros::ros-info
        "support-style: ~a, yaw-angle: ~a, gripper-angle: ~a, approach-pos: ~a, obj-moved-pos: ~a"
                     (gethash :support-style as-table)
                     (gethash :yaw-angle as-table)
                     (gethash :gripper-angle as-table)
                     (gethash :approach-pos as-table)
                     (gethash :obj-moved-pos as-table))
      (setq obj-local-pos (gethash :obj-moved-pos as-table))
      (ros::ros-info "[:pick-object-with-movable-region] arm:~a approach to the object" arm)
      (ros::ros-info "[:pick-object-with-movable-region] arm:~a obj-local-pos: ~a"
                     arm obj-local-pos)
      (send *ri* :gripper-servo-on arm)
      ;; Setup arm for picking
      (let (next-avs target-poss (approach-coords (send (elt movable-region 3) :copy-worldcoords)))
        ;; target-poss: worldpos
        (send *baxter* :rotate-gripper arm
              (rad2deg (gethash :gripper-angle as-table)) :relative nil)
        (send *baxter* :slide-gripper arm 0 :relative nil)
        (let ((target-local-pos (copy-object (gethash :approach-pos as-table))))
          ;; target-local-pos: relative to movable-region coords
          ;; movable-region: ((min-of-x max-of-x) (min-of-y max-of-y) (min-of-z max-of-z) center-coords)
          (setf (elt target-local-pos 2) (+ (cadr (elt movable-region 2)) 200))
          ;; location above of the target object in local z axis
          (pushback (send (elt movable-region 3) :transform-vector target-local-pos) target-poss)
          (setf (elt target-local-pos 2) (+ (cadr (elt movable-region 2)) 100))
          (pushback (send (elt movable-region 3) :transform-vector target-local-pos) target-poss)
          ;; Go to a little inside of the bin/tote
          ;; TODO: check if this is good for kiva
          (setf (elt target-local-pos 2) (cadr (elt movable-region 2)))
          (pushback (send (elt movable-region 3) :transform-vector target-local-pos) target-poss))
        (ros::ros-info "[:pick-object-with-movable-region] arm:~a yaw-angle ~a"
                       arm (gethash :yaw-angle as-table))
        (if (or (eq grasp-style :suction) (eq (gethash :support-style as-table) :scrape-out))
          ;; suction
          (progn
            (send approach-coords :rotate (or (gethash :yaw-angle as-table) 0) :z :local)
            (send approach-coords :rotate (gethash :gripper-angle as-table) :y :local)
            (setq approach-rpy (car (send approach-coords :rpy-angle)))
            (dolist (target-pos target-poss)
              (pushback
                (send *baxter* :rotate-wrist-ik arm
                      (make-coords :pos target-pos
                                   :rpy approach-rpy)
                      :move-palm-end nil
                      :rotation-axis (if (null (gethash :yaw-angle as-table)) :z t))
                next-avs))
            ;; Fold fingers to avoid collision to the bin walls
            (send *ri* :move-hand arm
                  (send *baxter* :hand-grasp-pre-pose arm
                        (if (eq (gethash :support-style as-table) :opposed)
                          :opposed :cylindrical))
                  1000 :wait nil)
            (send *ri* :move-hand arm
                  (send *baxter* :hand-grasp-pose arm
                        (if (eq (gethash :support-style as-table) :opposed)
                          :opposed :cylindrical)
                        :angle 90)
                  1000 :wait t))
          ;; pinch
          (progn
            (send approach-coords :rotate (gethash :yaw-angle as-table) :z :local)
            (setq approach-rpy (car (send approach-coords :rpy-angle)))
            (dolist (target-pos target-poss)
              (pushback
                (send *baxter* :rotate-wrist-ik arm
                      (make-coords :pos target-pos
                                   :rpy approach-rpy)
                      :move-palm-end t
                      :rotation-axis t)
                next-avs))
            (send *ri* :move-hand arm
                  (send *baxter* :hand-grasp-pose arm :cylindrical :angle 90)
                  1000)))
        (dolist (av next-avs)
          (unless av
            (ros::ros-error "[:pick-object-with-movable-region] arm:~a IK before trying grasp fails. Abort picking" arm)
            (return-from :pick-object-with-movable-region (list :ik-failed nil))))
        ;; Move whole arm to location above of the target pose
        (send *ri* :angle-vector-sequence-raw next-avs
              :fast (send *ri* :get-arm-controller arm :head t) 0 :scale 5.0)
        (send *ri* :wait-interpolation))
      ;; Now hand is inside of the bin, so we can open fingers
      (send *ri* :move-hand arm
            (send *baxter* :hand-grasp-pre-pose arm
                  (if (eq (gethash :support-style as-table) :opposed)
                    :opposed :cylindrical))
            1000)
      ;; Picking trials in the bin
      (dotimes (i n-trial)
        (dotimes (j n-trial-same-pos)
          (when (or (null pick-result) (eq pick-result :grasp-failed))
            (setq pick-result
                  (send self :try-to-pick-object arm
                        (make-coords
                          :pos (send (elt movable-region 3) :transform-vector obj-local-pos)
                          :rpy approach-rpy)
                        :offset (float-vector 0 0 (* i -30))
                        :grasp-style grasp-style
                        :support-style (gethash :support-style as-table)
                        :rotation-axis (if (null (gethash :yaw-angle as-table)) :z t)
                        :gripper-angle (gethash :gripper-angle as-table))))))
      (when do-stop-grasp
        (unless (or (eq pick-result :grasp-succeeded)
                    (eq pick-result :grasp-succeeded-with-support))
          (send *ri* :stop-grasp arm)
          (if (eq grasp-style :pinch)
            (send *ri* :stop-grasp arm :pinch))))
      (if (and (eq grasp-style :suction)
               (eq pick-result :grasp-succeeded))
        (progn
          ;; Move underactuated fingers to initial pose
          (send *ri* :move-hand arm
                (send *baxter* :hand-grasp-pre-pose arm :opposed) 1000 :wait nil)))
      (send *ri* :gripper-servo-on arm)
      (list pick-result (gethash :support-style as-table))))
  (:get-object-position
    (arm movable-region &key (object-index 0))
    (let (obj-box obj-coords obj-local-pos)
      (setq obj-box (elt (gethash arm object-boxes-) object-index))
      (setq obj-coords (elt (gethash arm object-coords-) object-index))
      ;; relative position to movable-region coords
      (setq obj-local-pos
            (send (send (elt movable-region 3) :transformation obj-coords :local) :pos))
      ;; TODO: fix this offset for camera point cloud
      (if (eq arm :rarm) (setf (aref obj-local-pos 2) (- (aref obj-local-pos 2) 18)))
      (dolist (i (list 0 1))
        (cond ((> (elt (elt movable-region i) 0) (aref obj-local-pos i))
               (ros::ros-error "[:get-object-position] object is out of movable region. ~a > ~a < ~a"
                               (elt (elt movable-region i) 1)
                               (aref obj-local-pos i)
                               (elt (elt movable-region i) 0))
               (setf (aref obj-local-pos i) (elt (elt movable-region i) 0)))
              ((> (aref obj-local-pos i) (elt (elt movable-region i) 1))
               (ros::ros-error "[:get-object-position] object is out of movable region. ~a < ~a > ~a"
                               (elt (elt movable-region i) 1)
                               (aref obj-local-pos i)
                               (elt (elt movable-region i) 0))
               (setf (aref obj-local-pos i) (elt (elt movable-region i) 1)))
              (t nil)))
      obj-local-pos))
  (:assoc-attached-object-to-end-coords
    (arm end-coords-symbol)
    (send *baxter* arm end-coords-symbol :assoc (gethash arm attached-objects-)))
  (:dissoc-attached-object-from-end-coords
    (arm end-coords-symbol)
    (send *baxter* arm end-coords-symbol :dissoc (gethash arm attached-objects-)))
  (:check-near-walls
    (obj-pos movable-region &key (range 100))
    (let (near-walls)
      (when (< (- (aref obj-pos 0) (elt (elt movable-region 0) 0)) range)
        (pushback :rear near-walls)
        (ros::ros-info-cyan "[:check-near-walls] Object is near rear wall"))
      (when (< (- (elt (elt movable-region 0) 1) (aref obj-pos 0)) range)
        (pushback :front near-walls)
        (ros::ros-info-cyan "[:check-near-walls] Object is near front wall"))
      (when (< (- (aref obj-pos 1) (elt (elt movable-region 1) 0)) range)
        (pushback :right near-walls)
        (ros::ros-info-cyan "[:check-near-walls] Object is near right wall"))
      (when (< (- (elt (elt movable-region 1) 1) (aref obj-pos 1)) range)
        (pushback :left near-walls)
        (ros::ros-info-cyan "[:check-near-walls] Object is near left wall"))
      near-walls))
  (:decide-approach-and-support
    (arm obj-local-pos obj-local-cube movable-region grasp-style)
    "Return value: hash table of :support-style, :yaw-angle, :gripper-angle,
    :approach-pos and :obj-moved-pos.
    support-style is nil, :cylindrical, :opposed or :scrape-out.
    yaw-angle is nil, -pi/2 ~ pi/2 or pi.
    gripper-angle is 0 or -pi/2.
    approach-pos is float-vector of local pos of passing point
    obj-moved-pos is float-vector of moved local pos of object
    CAUTION: This algorithm depends on PCA in ClusterPointIndicesDecomposer"
    (let (return-table support-style yaw-angle gripper-angle approach-pos obj-moved-pos
                       approach-candidates approach-idx
                       min-dist-bw-obj-bin
                       (region-coords (elt movable-region 3))
                       (gravity-vec #f(0 0 -1))
                       (gripper-req-l (+ (gethash arm (gethash :pad-link-l gripper-dims-))
                                         (/ (gethash arm (gethash :tube-t gripper-dims-)) 2))))
      ;; initialize approach-candidates
      ;; each approach candidate is (index of direction (x: 0, y: 1, z: 2), sign of direction)
      ;; (list 2 1) is impossible as that direction is from region center to entrance
      (setq approach-candidates
            (if (eq grasp-style :suction)
              (list (list)  ;; best candidate
                    ;; middle candidates
                    (list (list 0 1) (list 0 -1) (list 1 1) (list 1 -1))
                    ;; worst candidate in common as it pushes object to back of bin
                    (list (list 2 -1)))
              (list (list (list 2 -1))  ;; best candidate as scraping is unstable
                    ;; middle candidates
                    (list)
                    ;; worst candidate
                    (list (list 0 1) (list 0 -1) (list 1 1) (list 1 -1)))))
      ;; convert gravity vector from world to local
      (setq gravity-vec (send region-coords :inverse-transform-vector
                              (v+ gravity-vec (send region-coords :worldpos))))
      ;; decide best or middle approach direction based on gravity direction
      (let (max-idx max-sign (max-val 0))
        (dotimes (i 3)
          (when (> (abs (aref gravity-vec i)) max-val)
            (setq max-idx i)
            (setq max-val (abs (aref gravity-vec i)))
            (if (< (aref gravity-vec i) 0) (setq max-sign -1) (setq max-sign 1))))
        ;; remove unnecessary approach candidates
        (dotimes (i 3)
          (dolist (candidate (list (list max-idx max-sign)
                                   ;; direction opposite to gravity
                                   (list max-idx (- max-sign))))
            (setf (elt approach-candidates i)
                  (remove candidate (elt approach-candidates i) :test #'equal))))
        (pushback (list max-idx max-sign)
                  (elt approach-candidates (if (eq grasp-style :suction) 0 1))))
      ;; check which approach can be performed
      (dolist (candidate-list approach-candidates)
        (setq min-dist-bw-obj-bin *inf*)
        (dolist (candidate candidate-list)
          (let ((idx (elt candidate 0))
                (sign (elt candidate 1))
                (approach-coords (make-coords))
                (obj-approach-cube (copy-object obj-local-cube))
                vertex-values dist-bw-obj-bin opp-dist-bw-obj-bin object-t object-w object-h
                obj-pos-to-walls)
            ;; decide approach-coords (minus z is approach direction)
            (cond ((and (= idx 0) (< sign 0))
                   (send approach-coords :rotate pi/2 :y :local))
                  ((and (= idx 0) (> sign 0))
                   (send approach-coords :rotate pi :z :local)
                   (send approach-coords :rotate pi/2 :y :local))
                  ((and (= idx 1) (< sign 0))
                   (send approach-coords :rotate pi/2 :z :local)
                   (send approach-coords :rotate pi/2 :y :local))
                  ((and (= idx 1) (> sign 0))
                   (send approach-coords :rotate -pi/2 :z :local)
                   (send approach-coords :rotate pi/2 :y :local))
                  ((and (= idx 2) (< sign 0))
                   ;; decide yaw angle by longest side of projected obj cube
                   ;; use this angle to estimate precise box size for front suction decision
                   (let (sides vec proj-vec max-proj-vec)
                     (pushback (list (x-of-cube obj-local-cube)) sides)
                     (pushback (list (y-of-cube obj-local-cube)) sides)
                     (pushback (list (z-of-cube obj-local-cube)) sides)
                     (dotimes (i 3)
                       (setq vec (float-vector 0 0 0))
                       (setf (aref vec i) 1)
                       (setq vec (scale (car (elt sides i)) vec))
                       (setq proj-vec (send obj-local-cube :transform-vector vec))
                       (setq proj-vec (v- proj-vec (send obj-local-cube :worldpos)))
                       (setf (aref proj-vec 2) 0)
                       (pushback (copy-object proj-vec) (elt sides i)))
                     (sort sides #'(lambda (x y) (> (norm (cadr x)) (norm (cadr y)))))
                     ;; get object dimension for front suction decision
                     (setq object-t (car (elt sides 1)))
                     (setq object-w (car (elt sides 0)))
                     (setq object-h (car (elt sides 2)))
                     ;; decide yaw angle by longest side of projected obj cube
                     (setq max-proj-vec (normalize-vector (cadr (elt sides 0))))
                     (if (>= (aref max-proj-vec 0) 0)
                       (setq yaw-angle (- (acos (v. max-proj-vec #f(0 1 0)))))
                       (setq yaw-angle (acos (v. max-proj-vec #f(0 1 0)))))
                     (if (> yaw-angle pi/2)
                       (setq yaw-angle (- yaw-angle pi)))
                     (if (< yaw-angle -pi/2)
                       (setq yaw-angle (+ yaw-angle pi))))
                   (send approach-coords :rotate yaw-angle :z :local))
                  (t (ros::ros-fatal
                       "[:decide-approach-and-support] Gravity draws objects outside bin")
                     (return-from :decide-approach-and-support nil)))
            ;; transform cube
            (send obj-approach-cube :move-to
                  (send approach-coords :transformation
                        (send obj-approach-cube :worldcoords) :local) :world)
            ;; update vertices
            (send obj-approach-cube :worldcoords)
            ;; decide support-style and yaw-angle
            (if (and (= idx 2) (< sign 0))
              (progn
                (dotimes (i 2)
                  (pushback (- (elt (elt movable-region i) 1) (aref obj-local-pos i))
                            obj-pos-to-walls)
                  (pushback (- (aref obj-local-pos i) (elt (elt movable-region i) 0))
                            obj-pos-to-walls))
                (if (eq grasp-style :suction)
                  ;; suction
                  ;; check if object is big enough to suction or too low for side suction
                  ;; TODO: fix hard-coded value
                  (when (or (and (> object-t
                                    (+ (* (gethash arm (gethash :pad-r gripper-dims-)) 2) 30))
                                 (> object-w
                                    (+ (* (gethash arm (gethash :pad-r gripper-dims-)) 2) 30)))
                            (and (< object-h object-t) (< object-h object-w)))
                    (setq gripper-angle 0)
                    (setq obj-moved-pos (copy-object obj-local-pos))
                    (setq approach-idx idx)
                    ;; decide support style depending on object size
                    ;; yaw-angle is already set
                    ;; TODO: fix hard-coded value
                    (cond ((and (<= object-t
                                    (gethash arm (gethash :opposed-max-w gripper-dims-)))
                                (>= object-w
                                    (+ (gethash arm (gethash :pad-to-finger gripper-dims-)) 25)))
                           (setq support-style :opposed)
                           (if (>= yaw-angle 0)
                             (setq yaw-angle (- yaw-angle pi/2))
                             (setq yaw-angle (+ yaw-angle pi/2))))
                          ((and (<= object-w
                                    (gethash arm (gethash :opposed-max-w gripper-dims-)))
                                (>= object-t
                                    (+ (gethash arm (gethash :pad-to-finger gripper-dims-)) 25)))
                           (setq support-style :opposed))  ;; yaw-angle is as is
                          (t (setq support-style nil)
                             (setq yaw-angle nil)))
                    (when yaw-angle
                      ;; follow yaw-angle change
                      (send approach-coords :orient yaw-angle :z :world)
                      ;; change obj-approach-cube based on new yaw-angle
                      (send obj-approach-cube :move-to (send obj-local-cube :worldcoords) :world)
                      (send obj-approach-cube :move-to
                            (send approach-coords :transformation
                                  (send obj-approach-cube :worldcoords) :local) :world)
                      ;; update vertices
                      (send obj-approach-cube :worldcoords)
                      (setq obj-moved-pos
                            (send approach-coords :inverse-transform-vector obj-moved-pos))
                      (setq vertex-values (mapcar #'(lambda (x) (aref x 0))
                                                  (send obj-approach-cube :vertices)))
                      ;; fix obj-pos to pinch object
                      ;; TODO: fix hard-coded value
                      (if (> (elt obj-moved-pos 0)
                             (- (apply #'max vertex-values)
                                (+ (gethash arm (gethash :pad-to-finger gripper-dims-)) 20)))
                        (setf (elt obj-moved-pos 0)
                              (- (apply #'max vertex-values)
                                 (+ (gethash arm (gethash :pad-to-finger gripper-dims-)) 20))))
                      (setq obj-moved-pos
                            (send approach-coords :transform-vector obj-moved-pos))
                      ;; remake obj-pos-to-walls based on new obj-pos
                      (setq obj-pos-to-walls nil)
                      (dotimes (i 2)
                        ;; limit obj-pos inside movable-region
                        (if (> (aref obj-moved-pos i) (elt (elt movable-region i) 1))
                          (setf (aref obj-moved-pos i) (elt (elt movable-region i) 1)))
                        (if (< (aref obj-moved-pos i) (elt (elt movable-region i) 0))
                          (setf (aref obj-moved-pos i) (elt (elt movable-region i) 0)))
                        (pushback (- (elt (elt movable-region i) 1) (aref obj-moved-pos i))
                                  obj-pos-to-walls)
                        (pushback (- (aref obj-moved-pos i) (elt (elt movable-region i) 0))
                                  obj-pos-to-walls)))
                    (setq approach-pos (copy-object obj-moved-pos))
                    ;; check and avoid near walls
                    ;; TODO: fix hard coded value
                    (let (near-walls forced-yaw-angle)
                      (when (< (elt obj-pos-to-walls 0) 35)
                        (pushback :front near-walls))
                      (when (< (elt obj-pos-to-walls 1) 35)
                        (pushback :rear near-walls))
                      (when (< (elt obj-pos-to-walls 2) 35)
                        (pushback :left near-walls))
                      (when (< (elt obj-pos-to-walls 3) 35)
                        (pushback :right near-walls))
                      (if (< (length near-walls) 2)
                        (cond
                          ((eq (car near-walls) :rear) (setq forced-yaw-angle 0.0))
                          ((eq (car near-walls) :front) (setq forced-yaw-angle pi))
                          ((eq (car near-walls) :right) (setq forced-yaw-angle pi/2))
                          ((eq (car near-walls) :left) (setq forced-yaw-angle -pi/2)))
                        (cond
                          ((and (member :rear near-walls) (member :right near-walls))
                           (setq forced-yaw-angle (/ pi 4)))
                          ((and (member :rear near-walls) (member :left near-walls))
                           (setq forced-yaw-angle (/ -pi 4)))
                          ((and (member :front near-walls) (member :right near-walls))
                           (setq forced-yaw-angle (/ (* 3 pi) 4)))
                          ((and (member :front near-walls) (member :left near-walls))
                           (setq forced-yaw-angle (/ (* 3 -pi) 4)))))
                      (if (and yaw-angle forced-yaw-angle
                               (not (eps= yaw-angle (abs forced-yaw-angle) 0.1)))
                        (setq support-style nil))
                      (if forced-yaw-angle (setq yaw-angle forced-yaw-angle))))
                  ;; pinch
                  ;; TODO: consider object size instead of considering object centroid only
                  ;; TODO: fix hard coded value
                  (when
                    (= (length obj-pos-to-walls)
                       (length (remove nil (mapcar #'(lambda (x) (> x 50))
                                                   obj-pos-to-walls))))
                    ;; yaw-angle is already set
                    (setq support-style :cylindrical)
                    (setq gripper-angle 0)
                    (setq approach-pos (copy-object obj-local-pos))
                    (setq obj-moved-pos (copy-object obj-local-pos))
                    (setq approach-idx idx))))
              (let (sides vec proj-vec)
                (progn
                  (pushback (list (x-of-cube obj-approach-cube)) sides)
                  (pushback (list (y-of-cube obj-approach-cube)) sides)
                  (pushback (list (z-of-cube obj-approach-cube)) sides)
                  (dotimes (i 3)
                    (setq vec (float-vector 0 0 0))
                    (setf (aref vec i) 1)
                    (setq vec (scale (car (elt sides i)) vec))
                    (setq proj-vec (send obj-approach-cube :transform-vector vec))
                    (setq proj-vec (v- proj-vec (send obj-approach-cube :worldpos)))
                    (setf (aref proj-vec 2) 0)
                    (pushback (copy-object proj-vec) (elt sides i)))
                  (sort sides #'(lambda (x y) (> (norm (cadr x)) (norm (cadr y)))))
                  ;; get object dimension for support style decision
                  (setq object-t (car (elt sides 2))))
                (setq vertex-values (mapcar #'(lambda (x) (aref x 1))
                                            (send obj-approach-cube :vertices)))
                (setq object-w (- (apply #'max vertex-values) (apply #'min vertex-values)))
                (setq vertex-values (mapcar #'(lambda (x) (aref x 2))
                                            (send obj-approach-cube :vertices)))
                (setq dist-bw-obj-bin (- (apply #'min vertex-values)
                                         (elt (elt movable-region idx) 0)))
                (setq opp-dist-bw-obj-bin (- (elt (elt movable-region idx) 1)
                                             (apply #'max vertex-values)))
                ;; check if gripper can enter
                ;; TODO: dismiss concern about little point cloud in bin
                ;;       and add size check like above
                ;; TODO: manage standing flat object not parallel to bin wall
                ;; TODO: fix hard-coded variable
                (when (and (> opp-dist-bw-obj-bin gripper-req-l)
                           (< dist-bw-obj-bin min-dist-bw-obj-bin)
                           (> (car (elt sides 0))
                              (* (gethash arm (gethash :pad-r gripper-dims-)) 2))
                           (> (car (elt sides 1))
                              (* (gethash arm (gethash :pad-r gripper-dims-)) 2)))
                  ;; decide support style depending on object size
                  (setq gripper-angle -pi/2)
                  (cond ((eq grasp-style :pinch)
                         (setq support-style :scrape-out))
                        ((<= object-t (gethash arm (gethash :cylindrical-max-t gripper-dims-)))
                         (setq support-style :cylindrical))
                        ((<= object-w (gethash arm (gethash :opposed-max-w gripper-dims-)))
                         (setq support-style :opposed))
                        (t (setq support-style nil)))
                  ;; decide yaw-angle
                  (cond ((and (= idx 0) (< sign 0))
                         (setq yaw-angle pi))
                        ((and (= idx 0) (> sign 0))
                         (setq yaw-angle 0))
                        ((and (= idx 1) (< sign 0))
                         (setq yaw-angle -pi/2))
                        ((and (= idx 1) (> sign 0))
                         (setq yaw-angle pi/2)))
                  (setq obj-moved-pos (copy-object obj-local-pos))
                  ;; not to collide palm to walls
                  ;; TODO: fix hard-coded variable and refactor
                  (let ((cross-idx (if (= idx 0) 1 0)) cross-min cross-max)
                    (setq cross-min (+ (elt (elt movable-region cross-idx) 0) 40))
                    (setq cross-max (- (elt (elt movable-region cross-idx) 1) 40))
                    (cond ((> cross-min (aref obj-moved-pos cross-idx))
                           (setf (aref obj-moved-pos cross-idx) cross-min))
                          ((> (aref obj-moved-pos cross-idx) cross-max)
                           (setf (aref obj-moved-pos cross-idx) cross-max))))
                  (when (eq grasp-style :suction)
                    ;; not to catch fingers between object and suction tube in side suction
                    (setq obj-moved-pos
                          (send approach-coords :inverse-transform-vector obj-moved-pos))
                    (setq vertex-values (mapcar #'(lambda (x) (aref x 0))
                                                (send obj-approach-cube :vertices)))
                    ;; TODO: fix hard-coded variable
                    (if (> (elt obj-moved-pos 0) (+ (apply #'min vertex-values) 40))
                      (setf (elt obj-moved-pos 0) (+ (apply #'min vertex-values) 40)))
                    (setq obj-moved-pos
                          (send approach-coords :transform-vector obj-moved-pos)))
                  ;; don't fix obj-pos overly (because bbox is sometimes outside bin)
                  (if (> (aref obj-moved-pos 2) (- (elt (elt movable-region 2) 1) 80))
                    (setf (aref obj-moved-pos 2) (- (elt (elt movable-region 2) 1) 80)))
                  (setq approach-pos (copy-object obj-moved-pos))
                  (setq approach-pos
                        (send approach-coords :inverse-transform-vector approach-pos))
                  (setq vertex-values (mapcar #'(lambda (x) (aref x 2))
                                              (send obj-approach-cube :vertices)))
                  (setf (elt approach-pos 2) (apply #'max vertex-values))
                  (setq approach-pos
                        (send approach-coords :transform-vector approach-pos))
                  ;; limit palm position not to collide fingers to bin
                  ;; TODO: fix hard-coded variable
                  (dotimes (i 2)
                    (if (> (aref approach-pos i) (- (elt (elt movable-region i) 1) 30))
                      (setf (aref approach-pos i) (- (elt (elt movable-region i) 1) 30)))
                    (if (< (aref approach-pos i) (+ (elt (elt movable-region i) 0) 30))
                      (setf (aref approach-pos i) (+ (elt (elt movable-region i) 0) 30))))
                  (setq approach-idx idx)
                  (setq min-dist-bw-obj-bin dist-bw-obj-bin))))))
        (if approach-idx (return)))
      (if (null approach-idx)
        (progn
          (ros::ros-fatal "[:decide-approach-and-support] There is no possible grasping")
          nil)
        (progn
          (setq return-table (make-hash-table))
          (sethash :support-style return-table support-style)
          (sethash :yaw-angle return-table yaw-angle)
          (sethash :gripper-angle return-table gripper-angle)
          (sethash :approach-pos return-table approach-pos)
          (sethash :obj-moved-pos return-table obj-moved-pos)
          return-table))))
  (:try-to-pick-object
    (arm obj-coords &key (offset #f(0 0 0)) (grasp-style :suction) support-style
         (rotation-axis :z) (gripper-angle 0))
    (cond ((and (eq grasp-style :suction) (= gripper-angle 0))
           (send self :try-to-suction-front-object arm obj-coords support-style rotation-axis
                 :offset offset))
          ((eq grasp-style :suction)
           (send self :try-to-suction-side-object arm obj-coords support-style rotation-axis
                 :offset offset))
          ((eq support-style :scrape-out)
           (send self :try-to-scrape-out-object arm obj-coords :offset offset))
          (t
            (send self :try-to-pinch-object arm obj-coords :offset offset))))
  (:try-to-suction-object-with-gripper-v4
    (arm obj-pos &key (offset #f(0 0 0)))
    "Return value: :grasp-succeeded or :grasp-failed"
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
      (setq graspingp (send *ri* :graspingp arm :suction))
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
      (setq graspingp (send *ri* :graspingp arm :suction))
      (ros::ros-info "[:try-to-suction-object-with-gripper-v4] arm:~a graspingp: ~a" arm graspingp)
      (if graspingp :grasp-succeeded :grasp-failed)))
  (:try-to-suction-front-object
    (arm obj-coords support-style rotation-axis &key (offset #f(0 0 0)))
    "Return value: :grasp-failed, :grasp-succeeded or :grasp-succeeded-with-support"
    (let (graspingp av
          (coords-before-approach (send *baxter* arm :end-coords :copy-worldcoords)))
      ;; when comment in below, object is dropped from kiva
      ;; (send coords-before-approach :locate #f(0 0 300) obj-coords)
      ;; start the vacuum gripper before approaching to the object
      (ros::ros-info "[:try-to-suction-front-object] arm:~a start vacuum gripper" arm)
      (send *ri* :start-grasp arm)
      ;; suction: prismatic-based approach
      (send *baxter* :slide-gripper arm (send (send *baxter* arm :gripper-x) :max-angle) :relative nil)
      (send *baxter* arm :inverse-kinematics
            (send (send obj-coords :copy-worldcoords) :locate offset :local)
            :use-gripper t
            :rotation-axis rotation-axis)
      ;; assoc attached object to end-coords for stowing check
      (let (vertex-values)
        ;; update vertices
        (send (gethash arm attached-objects-) :worldcoords)
        (setq vertex-values
              (mapcar #'(lambda (x) (aref (send obj-coords :inverse-transform-vector x) 2))
                      (send (gethash arm attached-objects-) :vertices)))
        ;; make attached object contact with vacuum pad
        (send (gethash arm attached-objects-) :translate
              (v- offset (float-vector 0 0 (apply #'max vertex-values))) obj-coords)
        (send self :assoc-attached-object-to-end-coords arm :end-coords))
      (let ((prismatic-angle (send *baxter* arm :gripper-x :joint-angle)))
        (send *baxter* :slide-gripper arm 0 :relative nil)
        (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
              3000 (send *ri* :get-arm-controller arm) 0)
        (send *ri* :wait-interpolation)  ;; move down only the hand palm

        (send *baxter* :slide-gripper arm prismatic-angle :relative nil)
        (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
              3000 (send *ri* :get-arm-controller arm) 0)
        ;; FIXME: :wait-interpolation-until using :prismatic-loaded sometimes ends too fast,
        ;; so currently we only check :grasp (suction pressure).
        (send *ri* :wait-interpolation-until arm :grasp)

        (send *baxter* :slide-gripper arm (send (send *baxter* arm :gripper-x) :max-angle) :relative nil)  ;; maximum angle of prismatic joint
        (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
              3000 (send *ri* :get-arm-controller arm) 0)
        (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)
        )
      (setq graspingp (send *ri* :graspingp arm :suction))
      (ros::ros-info "[:try-to-suction-front-object] arm:~a graspingp: ~a" arm graspingp)
      (unless graspingp
        (ros::ros-info "[:try-to-suction-front-object] arm:~a again approach to the object" arm)
        (let ((temp-av (send *baxter* :angle-vector)))
          ;; only if robot can solve IK
          (if (send *baxter* arm :move-end-pos #f(0 0 -100) :local)
            (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
                  3000 (send *ri* :get-arm-controller arm) 0))
          (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)
          (send *ri* :angle-vector-raw (send *baxter* :angle-vector temp-av)
                3000 (send *ri* :get-arm-controller arm) 0) ;; revert baxter
          (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)))
      ;; free vacuum pad joint
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (send *ri* :angle-vector-raw (send *baxter* :rotate-gripper arm 30 :relative nil)
            1000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      ;; lift a little to world-z for drawing obj
      (send *ri* :angle-vector-raw (send *baxter* arm :move-end-pos #f(0 0 20) :world)
            1000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      ;; suction: prismatic-based approach
      (send *baxter* :slide-gripper arm 0 :relative nil)
      (send *baxter* :rotate-gripper arm 0 :relative nil)
      ;; suction: prismatic-based approach
      ;; lift object
      (ros::ros-info "[:try-to-suction-front-object] arm:~a lift the object" arm)
      (send *ri* :gripper-servo-off arm)
      (let ((tc (send *baxter* arm :end-coords :copy-worldcoords)) trans-vec)
        ;; overwrite only local-z
        (setq trans-vec
              (send tc :inverse-transform-vector (send coords-before-approach :worldpos)))
        (send tc :translate (float-vector 0 0 (aref trans-vec 2)) :local)
        (send *baxter* arm :inverse-kinematics tc :rotation-axis :z)
        )
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      (setq graspingp (send *ri* :graspingp arm :suction))
      (ros::ros-info "[:try-to-suction-front-object] arm:~a graspingp: ~a" arm graspingp)
      (if (null graspingp)
        :grasp-failed
        (if support-style
          (let (finger-av)
            (unix::sleep 1)
            (send *ri* :calib-proximity-threshold arm)
            (send *ri* :move-hand arm
                  (send *baxter* :hand-grasp-pose arm support-style) 1000 :wait nil)
            (send *ri* :wait-interpolation-until arm :hand :finger-proximity)
            ;; TODO: fix hard-coded value
            (setq finger-av (send *ri* :get-real-finger-av arm))
            (ros::ros-info "[:try-to-suction-front-object] arm:~a finger-av: ~a"
                           arm finger-av)
            (if (and (> (aref finger-av (- (length finger-av) 1)) 30)
                     (< (aref finger-av (- (length finger-av) 1)) 86))
              (progn
                (send *ri* :move-hand arm
                      (send *baxter* :hand-grasp-pose arm support-style) 1000)
                (ros::ros-info "[:try-to-suction-front-object] arm:~a supporting succeeded" arm)
                :grasp-succeeded-with-support)
              (progn
                (send *ri* :move-hand arm
                      (send *baxter* :hand-grasp-pre-pose arm support-style) 1000 :wait nil)
                (ros::ros-info "[:try-to-suction-front-object] arm:~a supporting failed" arm)
                :grasp-succeeded)))
          :grasp-succeeded))))
  (:try-to-suction-side-object
    (arm obj-coords support-style rotation-axis &key (offset #f(0 0 0)))
    "Return value: :grasp-failed, :grasp-succeeded or :grasp-succeeded-with-support"
    (let (graspingp gripper-angle obj->end obj-above-av
                    (obj-coords-offset (send obj-coords :copy-worldcoords))
                    (coords-before-approach (send *baxter* arm :end-coords :copy-worldcoords)))
      (send obj-coords-offset :locate offset :local)
      (let (av above-coords)
        (setq av (send *baxter* :angle-vector))
        (setq gripper-angle (send *baxter* arm :gripper-p :joint-angle))
        ;; change gripper angle not to push target object
        (send *baxter* :rotate-gripper arm 0 :relative nil)
        (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation)
        ;; reproduce angle-vector for IK
        (send *baxter* :angle-vector av)
        ;; make coords above object
        (setq obj->end (send obj-coords-offset :transformation coords-before-approach :local))
        (setq above-coords (send obj-coords-offset :copy-worldcoords))
        (send above-coords :translate
              (float-vector 0 0 (aref (send obj->end :pos) 2))
              :local)
        (send *baxter* :slide-gripper arm
              (send (send *baxter* arm :gripper-x) :max-angle) :relative nil)
        (setq obj-above-av (send *baxter* arm :inverse-kinematics above-coords
                                 :rotation-axis rotation-axis))
        ;; change gripper angle not to push target object
        (send *baxter* :rotate-gripper arm 0 :relative nil)
        ;; move gripper above object
        (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
              3000 (send *ri* :get-arm-controller arm) 0)
        (send *ri* :wait-interpolation)
        (unix::sleep 1)
        ;; reproduce gripper angle
        (send *baxter* :rotate-gripper arm (+ gripper-angle 10) :relative nil)
        (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation))
      ;; start the vacuum gripper before approaching to the object
      (ros::ros-info "[:try-to-suction-side-object] arm:~a start vacuum gripper" arm)
      (send *ri* :start-grasp arm)
      ;; approach to object
      ;; TODO: return if ik failure
      (send *baxter* arm :inverse-kinematics
            obj-coords-offset
            :rotation-axis rotation-axis)
      ;; assoc attached object to end-coords for stowing check
      (let (vertex-values)
        ;; update vertices
        (send (gethash arm attached-objects-) :worldcoords)
        (setq vertex-values
              (mapcar #'(lambda (x) (aref (send obj-coords :inverse-transform-vector x) 2))
                      (send (gethash arm attached-objects-) :vertices)))
        ;; make attached object contact with vacuum pad
        (send (gethash arm attached-objects-) :translate
              (v- offset (float-vector 0 0 (apply #'max vertex-values))) obj-coords)
        (send self :assoc-attached-object-to-end-coords arm :end-coords))
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
            1000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation-until arm :grasp)
      (setq graspingp (send *ri* :graspingp arm :suction))
      (ros::ros-info "[:try-to-suction-side-object] arm:~a graspingp: ~a" arm graspingp)
      (unless graspingp
        (ros::ros-info "[:try-to-suction-side-object] arm:~a again approach to the object" arm)
        (let ((temp-av (send *baxter* :angle-vector)))
          ;; only if robot can solve IK
          (if (send *baxter* arm :move-end-pos #f(0 0 -100) :local)
            (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
                  3000 (send *ri* :get-arm-controller arm) 0))
          (send *ri* :wait-interpolation-until arm :grasp)
          (send *ri* :angle-vector-raw (send *baxter* :angle-vector temp-av)
                3000 (send *ri* :get-arm-controller arm) 0) ;; revert baxter
          (send *ri* :wait-interpolation)
          ))
      ;; lift object
      (ros::ros-info "[:try-to-suction-side-object] arm:~a lift the object" arm)
      (send *baxter* :angle-vector obj-above-av)
      (send *ri* :angle-vector-raw (send *baxter* arm :move-end-pos #f(0 0 50) :world)
            1000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      ;; TODO: fix hard-coded value
      (send *baxter* :slide-gripper arm 40 :relative nil)
      ;(send *baxter* arm :move-end-pos
      ;      (float-vector (aref (send obj->end :pos) 0) (aref (send obj->end :pos) 1) 0)
      ;      (send obj-coords-offset :copy-worldcoords)
      ;      :rotation-axis rotation-axis)
      ;; reproduce gripper angle
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (send *baxter* arm :inverse-kinematics
            coords-before-approach
            :rotation-axis rotation-axis)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      (setq graspingp (send *ri* :graspingp arm :suction))
      (ros::ros-info "[:try-to-suction-side-object] arm:~a graspingp: ~a" arm graspingp)
      (if (null graspingp)
        :grasp-failed
        (if support-style
          (let (finger-av)
            (unix::sleep 1)
            (send *ri* :calib-proximity-threshold arm)
            (send *ri* :move-hand arm
                  (send *baxter* :hand-grasp-pose arm support-style) 1000 :wait nil)
            (send *ri* :wait-interpolation-until arm :hand :finger-proximity)
            ;; TODO: fix hard-coded value
            (setq finger-av (send *ri* :get-real-finger-av arm))
            (ros::ros-info "[:try-to-suction-side-object] arm:~a finger-av: ~a"
                           arm finger-av)
            (if (and (> (aref finger-av (- (length finger-av) 1)) 30)
                     (< (aref finger-av (- (length finger-av) 1)) 120))
              (progn
                (send *ri* :move-hand arm
                      (send *baxter* :hand-grasp-pose arm support-style) 1000)
                (ros::ros-info "[:try-to-suction-side-object] arm:~a supporting succeeded" arm)
                :grasp-succeeded-with-support)
              (progn
                (send *ri* :move-hand arm
                      (send *baxter* :hand-grasp-pre-pose arm support-style) 1000 :wait nil)
                (ros::ros-info "[:try-to-suction-side-object] arm:~a supporting failed" arm)
                :grasp-succeeded)))
          :grasp-succeeded))))
  (:try-to-scrape-out-object
    (arm obj-coords &key (offset #f(0 0 0)))
    "Return value: :grasp-succeeded, :grasp-failed or :ik-failed"
    (let (graspingp av gripper-angle obj->end above-coords
                    (obj-coords-offset (send obj-coords :copy-worldcoords))
                    (coords-before-approach (send *baxter* arm :end-coords :copy-worldcoords)))
      (send obj-coords-offset :locate offset :local)
      (setq av (send *baxter* :angle-vector))
      (setq gripper-angle (send *baxter* arm :gripper-p :joint-angle))
      ;; change gripper angle not to push target object
      (send *baxter* :rotate-gripper arm 0 :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000)
      (send *ri* :wait-interpolation)
      ;; reproduce angle-vector for IK
      (send *baxter* :angle-vector av)
      ;; make coords above object
      (setq obj->end (send obj-coords-offset :transformation coords-before-approach :local))
      (setq above-coords (send obj-coords-offset :copy-worldcoords))
      (send above-coords :translate
            (float-vector 0 0 (aref (send obj->end :pos) 2))
            :local)
      (send *baxter* :slide-gripper arm
            (norm (send coords-before-approach :inverse-transform-vector
                        (send above-coords :worldpos)))
            :relative nil)
      (setq av
            (send *baxter* arm :inverse-kinematics above-coords
                  :rotation-axis t))
      (if (null av)
        (setq av
              (send *baxter* arm :inverse-kinematics above-coords
                    :rotation-axis :z)))
      (unless av
        (ros::ros-error
          "[:try-to-scrape-out-object] arm:~a IK to object fails. Abort picking"
          arm)
        (return-from :try-to-scrape-out-object :ik-failed))
      ;; change gripper angle not to push target object
      (send *baxter* :rotate-gripper arm 0 :relative nil)
      ;; move gripper above object
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      ;; push object to wall
      (send *baxter* arm :move-end-pos
            (float-vector 0 0 (- (+ (aref (send obj->end :pos) 2)
                                    (gethash arm (gethash :pad-link-l gripper-dims-)))))
            (send obj-coords-offset :copy-worldcoords))
      (send *baxter* :rotate-gripper arm 0 :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
            2000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation)
      ;; reproduce gripper angle to hang object
      ;; not to move down arm, send command to gripper only.
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
            1000 (if (eq arm :rarm) :rgripper-controller :lgripper-controller) 0)
      (send *ri* :wait-interpolation)
      ;; start the vacuum gripper after hanging object
      (ros::ros-info "[:try-to-scrape-out-object] arm:~a start vacuum gripper" arm)
      (send *ri* :start-grasp arm)
      ;; lift object
      (ros::ros-info "[:try-to-scrape-out-object] arm:~a lift the object" arm)
      (send *baxter* :slide-gripper arm 0 :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000)
      (send *ri* :wait-interpolation)
      ;; assoc attached object to end-coords for stowing check
      (let (vertex-values)
        (send (gethash arm attached-objects-) :locate
              (send *baxter* arm :palm-endpoint :worldpos) :world)
        ;; update vertices
        (send (gethash arm attached-objects-) :worldcoords)
        (setq vertex-values
              (mapcar #'(lambda (x) (aref (send (send *baxter* arm :palm-endpoint)
                                                :inverse-transform-vector x) 2))
                      (send (gethash arm attached-objects-) :vertices)))
        ;; make attached object contact with palm
        (send (gethash arm attached-objects-) :translate
              (v- (float-vector 0 0 (apply #'max vertex-values)))
              (send *baxter* arm :palm-endpoint))
        (send self :assoc-attached-object-to-end-coords arm :palm-endpoint))
      ;; close fingers
      (send *ri* :move-hand arm
            (send *baxter* :hand-grasp-pose arm :cylindrical) 1000 :wait nil)
      (send *baxter* arm :move-end-pos
            (float-vector (aref (send obj->end :pos) 0) (aref (send obj->end :pos) 1) 0)
            (send obj-coords-offset :copy-worldcoords))
      ;(send *baxter* arm :inverse-kinematics
      ;      coords-before-approach
      ;      :rotation-axis t)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation-smooth 1000)
      (setq graspingp (send *ri* :graspingp arm :pinch))
      (ros::ros-info "[:try-to-scrape-out-object] arm:~a graspingp: ~a" arm graspingp)
      (if graspingp :grasp-succeeded :grasp-failed)))
  (:try-to-pinch-object
    (arm obj-coords &key (offset #f(0 0 0)))
    "Return value: :grasp-succeeded, :grasp-failed or :ik-failed"
    (let (av graspingp pre-coords)
      ;; Initialize finger
      (send *ri* :move-hand arm
            (send *baxter* :hand-grasp-pose arm :cylindrical :angle 40) 1000)
      ;; start the vacuum gripper before approaching to the object
      (ros::ros-info "[:try-to-pinch-object] arm:~a start vacuum gripper" arm)
      (send *ri* :start-grasp arm)
      (setq av
            (send *baxter* arm :inverse-kinematics
                  (send (send obj-coords :copy-worldcoords) :locate offset :local)
                  :move-palm-end t
                  :rotation-axis t))
      (if (null av)
        (setq av
              (send *baxter* arm :inverse-kinematics
                    (send (send obj-coords :copy-worldcoords) :locate offset :local)
                    :move-palm-end t
                    :rotation-axis :z)))
      (unless av
        (ros::ros-error "[:try-to-pinch-object] arm:~a IK to object fails. Abort picking" arm)
        (return-from :try-to-pinch-object :ik-failed))
      ;; assoc attached object to end-coords for stowing check
      (let (vertex-values)
        ;; update vertices
        (send (gethash arm attached-objects-) :worldcoords)
        (setq vertex-values
              (mapcar #'(lambda (x) (aref (send obj-coords :inverse-transform-vector x) 2))
                      (send (gethash arm attached-objects-) :vertices)))
        ;; make attached object contact with palm
        (send (gethash arm attached-objects-) :translate
              (v- offset (float-vector 0 0 (apply #'max vertex-values))) obj-coords)
        (send self :assoc-attached-object-to-end-coords arm :palm-endpoint))
      (send *ri* :update-robot-state :wait-until-update t)
      (unless (send *ri* :angle-vector-raw av
                    3000 (send *ri* :get-arm-controller arm) 0 :end-coords-interpolation t)
        (send *ri* :angle-vector-raw av 3000 (send *ri* :get-arm-controller arm) 0))
      ;; Wait until grasp or finger touch
      (send *ri* :wait-interpolation-until arm
            :grasp :finger-flexion :finger-loaded :prismatic-loaded :finger-proximity)
      (setq av
            (send *baxter* arm :inverse-kinematics
                  obj-coords
                  :move-palm-end t
                  :rotation-axis t))
      (if (null av)
        (setq av
              (send *baxter* arm :inverse-kinematics
                    obj-coords
                    :move-palm-end t
                    :rotation-axis :z)))
      (unless av
        (ros::ros-error "[:try-to-pinch-object] arm:~a IK to object fails. Abort picking" arm)
        (return-from :try-to-pinch-object :ik-failed))
      (send *ri* :angle-vector-raw av 3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation-until arm
            :grasp :finger-flexion :finger-loaded :prismatic-loaded :finger-proximity)
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (send *baxter* arm :move-end-pos #f(0 0 -20) (send *baxter* arm :palm-endpoint))
      (send *ri* :angle-vector-raw
            (send *baxter* :slide-gripper arm
                  (send (send *baxter* arm :gripper-x) :max-angle) :relative nil)
            3000 (send *ri* :get-arm-controller arm) 0)
      (send *ri* :wait-interpolation-until arm :grasp :prismatic-loaded)
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (send *ri* :stop-grasp arm)
      (send *ri* :start-grasp arm :pinch)
      (send *ri* :start-grasp arm)
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (setq graspingp (send *ri* :graspingp arm :pinch))
      (ros::ros-info "[:try-to-pinch-object] arm:~a graspingp: ~a" arm graspingp)
      ;; lift object
      (ros::ros-info "[:try-to-pinch-object] arm:~a lift the object" arm)
      (setq pre-coords (send (send *baxter* arm :palm-endpoint) :copy-worldcoords))
      (send pre-coords :translate #f(0 0 200) :local)
      (setq av (send *baxter* arm :inverse-kinematics pre-coords
                     :move-palm-end t :rotation-axis t))
      (if (null av)
        (setq av
              (send *baxter* arm :inverse-kinematics pre-coords
                    :move-palm-end t :rotation-axis :z)))
      (send *ri* :angle-vector-raw av 3000 (send *ri* :get-arm-controller arm :gripper nil) 0)
      (send *ri* :wait-interpolation)
      ;; slide gripper to grasp tightly
      (send *baxter* :rotate-gripper arm -90 :relative nil)
      (send *baxter* :slide-gripper arm 0 :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector)
            :fast (send *ri* :get-arm-controller arm) 0 :scale 3.0)
      (send *ri* :wait-interpolation)
      (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (setq graspingp (send *ri* :graspingp arm :pinch))
      (ros::ros-info "[:try-to-pinch-object] arm:~a graspingp: ~a" arm graspingp)
      (if graspingp :grasp-succeeded :grasp-failed)))
  (:ik->cube-center
    (arm cube &key (offset #f(0 0 0)) (rpy #f(0 0 0)) (rotation-axis t)
         (use-gripper nil) (move-palm-end nil) (use-rotate-wrist-ik nil) (parallel-with :local))
    (let (cube-coords)
      (setq cube-coords (send cube :copy-worldcoords))
      (send cube-coords :translate
            (float-vector 0.0 0.0 (/ (z-of-cube cube) 2.0))
            :local)
      (send cube-coords :translate offset :local)
      (when (eq parallel-with :world)
        (setq cube-coords (make-coords :pos (send cube-coords :worldpos))))
      (send cube-coords :rotate (aref rpy 0) :z)
      (send cube-coords :rotate (aref rpy 1) :y)
      (send cube-coords :rotate (aref rpy 2) :x)
      (if use-rotate-wrist-ik
        (send *baxter* :rotate-wrist-ik arm cube-coords
              :rotation-axis rotation-axis
              :use-gripper use-gripper
              :move-palm-end move-palm-end)
        (send *baxter* arm :inverse-kinematics cube-coords
              :rotation-axis rotation-axis
              :use-gripper use-gripper
              :move-palm-end move-palm-end))))
  (:ik->bin-center
    (arm bin &rest args)
    (send* self :ik->cube-center arm (gethash bin bin-cubes-) args))
  (:ik->cardboard-center
    (arm cardboard &rest args)
    ;; use prismatic joint as much as possible
    (let ((jnt (send *baxter* arm :gripper-x)))
      (send jnt :joint-angle (send jnt :max-angle)))
    (send* self :ik->cube-center arm (gethash cardboard cardboard-cubes-) args))
  (:ik->tote-center
    (arm &rest args)
    (send* self :ik->cube-center arm (gethash arm tote-cubes-) args))
  (:move-arm-body->bin-overlook-pose
    (arm bin &key (gripper-angle 90))
    (let (avs offset rpy
              (offset-x -100)
              (offset-y (if (eq arm :larm) -50 50)))
      (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
      ;; (send *baxter* :reset-pose arm)
      (send *baxter* :avoid-shelf-pose arm bin)
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      (setq offset (float-vector offset-x offset-y 200))
      (setq rpy (float-vector 0 pi/2 0))
      (send self :ik->bin-center arm bin
            :offset offset :rpy rpy :use-gripper nil :use-rotate-wrist-ik t)
      ;; TODO: tilting gripper based on gravity and entrance direction
      (setq rpy (float-vector 0 (+ pi/2 (/ pi 6)) 0))
      (send self :ik->bin-center arm bin
            :offset offset :rpy rpy :use-gripper nil :use-rotate-wrist-ik t)
      (send *ri* :angle-vector (send *baxter* :angle-vector) 3000
            (send *ri* :get-arm-controller arm) 0)))
  (:move-arm-body->tote-overlook-pose
    (arm &key (gripper-angle 90))
    (let (avs offset rpy
              (offset-x (if (eq arm :larm) 0 -20))
              (offset-y (if (eq arm :larm) 390 -320)))
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
  (:set-target-location (arm location)
    (if (consp location)
      (setq location (format nil "~a_~a" (symbol2str (car location)) (symbol-string (cdr location))))
      (setq location (symbol2str location)))
    (ros::set-dynparam
      (format nil "/~a_hand_camera/candidates_publisher" (arm2str arm))
      (cons "target_location" location)))
  (:set-arm-state-param (arm state)
    (ros::set-param (format nil "~a_hand/state" (arm2str arm)) (symbol2str state)))
  (:get-bin-contents (bin)
    (ros::get-param
      (format nil "/bin_contents/~A" (symbol-string bin))))
  (:add-labeled-obj-scene (obj str-label)
    (let ((base-name (send (send *baxter* :base_lk) :name)))
      (send *co* :add-object obj :frame-id base-name
            :relative-pose (send obj :copy-worldcoords)
            :object-id str-label)))
  (:add-workspace-scene
    ()
    (let ((dimensions (list '(2500 10 2000) '(2500 10 2000)))
          (positions (list (float-vector 500 1200 0) (float-vector 500 -1200 0)))
          (names (list "workspace_left" "workspace_right")))
      (dotimes (i (length positions))
        (let ((cube (apply #'make-cube (elt dimensions i)))
              (pos (elt positions i))
              (name (elt names i)))
          (send cube :locate pos :world)
          (send self :add-labeled-obj-scene cube name)
          )
        )
      ))
  (:add-bin-scene (bin)
    (let ((cube (gethash bin bin-cubes-)))
      (send self :add-labeled-obj-scene cube
            (format nil "bin_~A" (symbol-string bin)))))
  (:delete-bin-scene (bin)
    (send *co* :delete-object (gethash bin bin-cubes-)))
  (:add-shelf-scene ()
    (dolist (bin bin-list-)
      (send self :add-bin-scene bin)))
  (:delete-shelf-scene ()
    (dolist (bin bin-list-)
      (send self :delete-bin-scene bin)))
  (:add-cardboard-scene (cardboard)
    (let (cardboard-cube)
      (setq cardboard-cube (gethash cardboard cardboard-cubes-))
      (send self :add-labeled-obj-scene cardboard-cube
            (format nil "cardboard_~A" (symbol-string cardboard)))))
  (:delete-cardboard-scene (cardboard)
    (let (cardboard-cube)
      (setq cardboard-cube (gethash cardboard cardboard-cubes-))
      (send *co* :delete-object cardboard-cube)))
  (:add-cardboad-rack-leg-scene (side)
    (let* ((cardboard-c-cube (gethash :c cardboard-cubes-))
           (cardboard-c-coords (send cardboard-c-cube :copy-worldcoords))
           rack-leg-cube offset)
      (setq offset
            (float-vector
              (elt (send cardboard-c-cube :worldpos) 0)
              (* (if (eq side :right) -1 1) (/ (y-of-cube cardboard-c-cube) 2.0))
              (elt (send cardboard-c-coords :worldpos) 2)))
      (setq rack-leg-cube (make-cube 300 20 1200))
      (send rack-leg-cube :translate offset :world)
      (send self :add-labeled-obj-scene rack-leg-cube
            (format nil "rack_leg_~a" (symbol2str side)))
      (sethash side rack-leg-cubes- rack-leg-cube)))
  (:delete-cardboard-rack-leg-scene (side)
    (send *co* :delete-object (gethash side rack-leg-cubes-)))
  (:add-cardboard-rack-scene ()
    (dolist (cardboard cardboard-list-)
      (send self :add-cardboard-scene cardboard))
    (dolist (side (list :left :right))
      (send self :add-cardboad-rack-leg-scene side)))
  (:delete-cardboard-rack-scene ()
    (dolist (cardboard cardboard-list-)
      (send self :delete-cardboard-scene cardboard))
    (dolist (side (list :left :right))
      (send self :delete-cardboad-rack-leg-scene side)))
  (:add-tote-scene (arm)
    (dolist (tmp-arm (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (let ((cube (gethash tmp-arm tote-cubes-)))
        (send self :add-labeled-obj-scene cube
              (format nil "~A_tote" (arm2str tmp-arm))))))
  (:delete-tote-scene (arm)
    (dolist (tmp-arm (if (eq arm :arms) (list :rarm :larm) (list arm)))
      (send *co* :delete-object (gethash tmp-arm tote-cubes-))))
  (:add-object-in-hand-scene (arm target-obj)
    ;; add object as attached collision object
    (let (end-coords-name obj-cube)
      ;; attach largest bounding box
      (setq obj-cube (gethash arm attached-objects-))
      (if (eq (send obj-cube :parent) (send *baxter* arm :end-coords))
        (setq end-coords-name (format nil "~a_gripper_pad_with_base" (arm2str arm)))
        (setq end-coords-name (format nil "~a_gripper_palm_endpoint" (arm2str arm))))
      (send *co* :add-attached-object obj-cube end-coords-name
            :frame-id end-coords-name
            :object-id target-obj
            :relative-pose (make-coords :pos (float-vector 0 0 (/ (z-of-cube obj-cube) 2)))
            :touch-links (list
                           (format nil "~a_gripper_base" (arm2str arm))
                           (format nil "~a_gripper_tube" (arm2str arm))
                           (format nil "~a_gripper_pad_with_base" (arm2str arm))
                           (format nil "~a_gripper_l_finger_base" (arm2str arm))
                           (format nil "~a_gripper_r_finger_base" (arm2str arm))
                           (format nil "~a_gripper_l_finger" (arm2str arm))
                           (format nil "~a_gripper_r_finger" (arm2str arm))
                           (format nil "~a_gripper_palm_endpoint" (arm2str arm))))))
  (:delete-object-in-hand-scene (arm)
    ;; delete object as attached collision object
    (let (obj-cube)
      (setq obj-cube (gethash arm attached-objects-))
      (send *co* :delete-attached-object obj-cube)
      (send *co* :delete-object obj-cube)))
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
      (cond ((consp src)
             (setq src (format nil "~a_~a" (symbol2str (car src)) (symbol-string (cdr src)))))
            ((symbolp src) (setq src (symbol2str src)))
            ((stringp src) nil)
            (t (ros::ros-fatal "[:update-json] invaild src")
               (return-from :update-json nil)))
      (cond ((consp dst)
             (setq dst (format nil "~a_~a" (symbol2str (car dst)) (symbol-string (cdr dst)))))
            ((symbolp dst) (setq dst (symbol2str dst)))
            ((stringp dst) nil)
            (t (ros::ros-fatal "[:update-json] invaild dst")
               (return-from :update-json nil)))
      (send req :src src)
      (send req :dst dst)
      (send req :item object)
      (send (ros::service-call service-name req) :updated)))
  (:save-json ()
    (let ((service-name (format nil "/json_saver/save_json"))
          (req (instance std_srvs::TriggerRequest :init)))
      (ros::wait-for-service service-name)
      (send (ros::service-call service-name req) :success)))
  (:get-scale-candidates (arm action &key (stamp (ros::time-now)))
    (let ((candidates-topic
            (format nil "~a_hand_camera/weight_candidates_refiner/output/candidates/~a"
                    (arm2str arm) (symbol2str action)))
          (changed-topic
            (format nil "~a_hand_camera/weight_candidates_refiner/output/changed_from_reset"
                    (arm2str arm)))
          candidates-msg changed-msg)
      (ros::subscribe candidates-topic jsk_recognition_msgs::LabelArray
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq candidates-msg msg)))))
      (ros::subscribe changed-topic jsk_recognition_msgs::BoolStamped
                      #'(lambda (msg)
                          (let ((st (send msg :header :stamp)))
                            (when (> (send st :to-sec) (send stamp :to-sec))
                              (setq changed-msg msg)))))
      (while (and (not (and candidates-msg changed-msg))
                  (> (+ (send stamp :to-sec) 10)
                     (send (ros::time-now) :to-sec)))
        (unix::usleep (* 50 1000))
        (ros::spin-once))
      (ros::unsubscribe candidates-topic)
      (ros::unsubscribe changed-topic)

      (if (and candidates-msg changed-msg)
        (if (send changed-msg :data)
          (mapcar #'(lambda (label-msg) (send label-msg :name)) (send candidates-msg :labels))
          :no-change)
        :timeout)))
  (:reset-scale (arm)
    (let ((service-name (format nil "/~a_hand_camera/weight_candidates_refiner/reset" (arm2str arm)))
          (req (instance std_srvs::TriggerRequest :init)))
      (ros::wait-for-service service-name)
      (send (ros::service-call service-name req) :success)))
  (:update-object-tables (obj bin &key object-type)
    (cond
      ((eq object-type :unfinished-objects)
       (sethash bin unfinished-objects-
                (remove-duplicates (append (gethash bin unfinished-objects-) (list obj))
                                   :test #'string=)))
      ((eq object-type :postponed-objects)
       (sethash bin unfinished-objects-
                (remove-duplicates (remove obj (gethash bin unfinished-objects-) :test #'string=)
                                   :test #'string=))
       (sethash bin postponed-objects-
                (remove-duplicates (append (gethash bin postponed-objects-) (list obj))
                                   :test #'string=)))
      ((eq object-type :finished-objects)
       ;; first, add into postponed-objects-
       (send self :update-object-tables obj bin :object-type :postponed-objects)
       ;; then move to finished-objects-
       ;; this is for the case that targets are same object.
       (sethash bin postponed-objects-
                (remove-duplicates (remove obj (gethash bin postponed-objects-) :test #'string=)
                                   :test #'string=))
       (sethash bin finished-objects-
                (remove-duplicates (append (gethash bin finished-objects-) (list obj))
                                   :test #'string=)))
      (t nil)))
  (:set-objects-param (arm bin &key object-type)
    (cond
      ((eq object-type :unfinished-objects)
       (ros::set-param
         (format nil "/~a_hand/unfinished_objects/~a" (arm2str arm)
                 (if (eq bin :tote) (symbol2str bin) (format nil "bin_~a" (symbol-string bin))))
         (gethash bin unfinished-objects-)))
      ((eq object-type :finished-objects)
       (ros::set-param
         (format nil "/~a_hand/finished_objects/~a" (arm2str arm)
                 (if (eq bin :tote) (symbol2str bin) (format nil "bin_~a" (symbol-string bin))))
         (gethash bin finished-objects-)))
      ((eq object-type :postponed-objects)
       (ros::set-param
         (format nil "/~a_hand/postponed_objects/~a" (arm2str arm)
                 (if (eq bin :tote) (symbol2str bin) (format nil "bin_~a" (symbol-string bin))))
         (gethash bin postponed-objects-)))
      ((eq object-type :all)
       (send self :set-objects-param arm bin :object-type :unfinished-objects)
       (send self :set-objects-param arm bin :object-type :finished-objects)
       (send self :set-objects-param arm bin :object-type :postponed-objects))
      (t nil)))
  (:reset-object-tables (location-type)
    (cond
      ((eq location-type :bins)
       (when (eq (length orders-) 0)
         (ros::ros-error "[:set-unfinished-objects] There is no order, :get-work-orders first.")
         (return-from :set-unfinished-objects nil))
       (dolist (order orders-)
         (let ((bin (str2symbol (send order :bin)))
               (obj (send order :item)))
           (send self :update-object-tables obj bin :object-type :unfinished-objects)))
       (dolist (bin bin-list-)
         (sethash bin finished-objects- nil)
         (sethash bin postponed-objects- nil)))
      ((eq location-type :tote)
       (let (tote-contents)
         (setq tote-contents (ros::get-param "/tote_contents"))
         (sethash :tote unfinished-objects- tote-contents)
         (sethash :tote finished-objects- nil)
         (sethash :tote postponed-objects- nil)))
      (t nil)))
  (:add-finished-object (arm obj bin)
    (send self :update-object-tables obj bin :object-type :finished-objects)
    (send self :set-objects-param arm bin :object-type :unfinished-objects)
    (send self :set-objects-param arm bin :object-type :postponed-objects)
    (send self :set-objects-param arm bin :object-type :finished-objects))
  (:add-postponed-object (arm obj bin)
    (send self :update-object-tables obj bin :object-type :postponed-objects)
    (send self :set-objects-param arm bin :object-type :unfinished-objects)
    (send self :set-objects-param arm bin :object-type :postponed-objects))
  (:get-grasp-style (arm obj trial-time)
    (let ((service-name
           (format nil "/grasp_style_server/get_grasp_style"))
          req res succeeded)
      (ros::wait-for-service service-name)
      (setq req (instance jsk_arc2017_common::GetGraspStyleRequest :init))
      (send req :item obj)
      (send req :trial_time trial-time)
      (setq res (ros::service-call service-name req))
      (setq succeeded (send res :success))
      (if succeeded
        (str2symbol (send res :style))
        (progn
          (ros::ros-warn "[:get-grasp-style] arm: ~a, cannot get grasp-style" (arm2str arm))
          (ros::ros-warn "[:get-grasp-style] message: ~a" (send res :message))
          succeeded))))
  (:decide-bin-return-area (arm bin)
    (let (bin-cube region region-x index)
      (setq bin-cube (gethash bin bin-cubes-))
      (setq region (send self :cube->movable-region bin-cube))
      (setq region-x (- (cadr (elt region 0)) (car (elt region 0))))
      (setq index (random 2))
      (cond
       ((eq index 0)
        (float-vector (/ (- region-x) 5.0)
                      0.0
                      0.0))
       ((eq index 1)
        (float-vector (/ region-x 5.0)
                      0.0
                      0.0)))))
  (:decide-tote-return-area (arm)
    (let (tote-cube region region-y)
      (setq tote-cube (gethash arm tote-cubes-))
      (setq region (send self :cube->movable-region tote-cube))
      (setq region-y (- (cadr (elt region 1)) (car (elt region 1))))
      (if (eq arm :larm)
        (float-vector 0 (/ region-y 8.0) 0)
        (float-vector 0 (/ region-y -8.0) 0))))
  (:decide-bin-place-area (arm bin grasp-style)
    (let (bin-cube region x-range index)
      (setq bin-cube (gethash bin bin-cubes-))
      (setq region (send self :cube->movable-region bin-cube))
      (setq x-range (- (cadr (elt region 0)) (car (elt region 0))))
      (if (eq grasp-style :suction)
        (let ((grav-vec (v+ (send bin-cube :worldpos) #f(0 0 -1))))
          (setq grav-vec (send bin-cube :inverse-transform-vector grav-vec))
          (if (> (abs (aref grav-vec 0)) 0.5)
            (progn
              (float-vector (- (/ x-range 2.0)
                               (+ (gethash arm (gethash :pad-link-l gripper-dims-))
                                  (/ (gethash arm (gethash :tube-t gripper-dims-)) 2))
                               30) 0 0))
            (progn
              (setq index (random 3))
              (cond
                ((eq index 0)
                 (float-vector 0 0 0))
                ((eq index 1)
                 (float-vector (/ x-range 5.0) 0 0))
                ((eq index 2)
                 (float-vector (/ x-range -5.0) 0 0))))))
        (float-vector 0 0 0))))
  (:decide-cardboard-place-area (arm cardboard)
    (let (cardboard-cube region region-y)
      (setq cardboard-cube (gethash cardboard cardboard-cubes-))
      (setq region (send self :cube->movable-region cardboard-cube))
      (setq region-y (- (cadr (elt region 1)) (car (elt region 1))))
      (if (= (random 2) 0)
        (float-vector -50.0 (/ region-y -7.0) 0.0)
        (float-vector -50.0 (/ region-y 7.0) 0.0))))
  (:get-stowing-bin-candidates (arm)
    (let (bins)
      (dotimes (i (length bin-list-))
        ;; TODO: check collision in robot pose by IK
        ;; (if (or (and (or (eq (elt bin-list- i) :a) (eq (elt bin-list- i) :d)) (eq arm :larm))
        ;;         (and (or (eq (elt bin-list- i) :c) (eq (elt bin-list- i) :f)) (eq arm :rarm)))
        (if (or ;(eq (mod i 3) 1)
                (and (eq (mod i 3) 0) (eq arm :larm))
                (and (eq (mod i 3) 2) (eq arm :rarm)))
          (pushback (elt bin-list- i) bins)))
      (sort bins #'<= #'(lambda (bin) (gethash bin bin-object-volumes-)))
      bins))
  (:update-bin-object-volumes (arm bin)
    (let (obj-bbox obj-bbox-volume)
      (setq obj-bbox (elt (gethash arm object-boxes-) 0))
      (setq obj-bbox-volume (* (send obj-bbox :dimensions :x)
                               (send obj-bbox :dimensions :y)
                               (send obj-bbox :dimensions :z)))
      (sethash bin bin-object-volumes-
               (+ (gethash bin bin-object-volumes-) obj-bbox-volume))))
  (:attached-obj-protruded-axes (arm place-target-cube)
    (let (attached-obj-local vertex-values vertices-in-entrance prot-axes)
      (setq attached-obj-local (copy-object (gethash arm attached-objects-)))
      ;; transform object
      (send attached-obj-local :move-to
            (send place-target-cube :transformation
                  (send attached-obj-local :worldcoords) :local) :world)
      ;; update vertices
      (send attached-obj-local :worldcoords)
      ;; check vertices to avoid body* error
      (setq vertex-values (mapcar #'(lambda (x) (aref x 2))
                                  (send attached-obj-local :vertices)))
      (if (find t (mapcar #'(lambda (x)
                              (< (abs (- (/ (z-of-cube place-target-cube) 2) x)) 3))
                          vertex-values))
        ;; body* will fail
        (progn
          (dotimes (i (length vertex-values))
            (when (< (elt vertex-values i) (+ (/ (z-of-cube place-target-cube) 2) 1))
              (pushback (elt (send attached-obj-local :vertices) i) vertices-in-entrance))))
        ;; body* will success
        (let (part-in-entrance)
          ;; get part of attached object inside entrance of place target
          (setq part-in-entrance
                (body* attached-obj-local (make-cube (* (x-of-cube place-target-cube) 3)
                                                     (* (y-of-cube place-target-cube) 3)
                                                     (z-of-cube place-target-cube))))
          ;; update vertices
          (send part-in-entrance :worldcoords)
          (setq vertices-in-entrance (send part-in-entrance :vertices))))
      ;; check which axis will protrude
      (setq vertex-values (mapcar #'(lambda (x) (aref x 0)) vertices-in-entrance))
      (when (or (> (apply #'max vertex-values) (/ (x-of-cube place-target-cube) 2))
                (< (apply #'min vertex-values) (/ (x-of-cube place-target-cube) -2)))
        (pushback :x prot-axes))
      (setq vertex-values (mapcar #'(lambda (x) (aref x 1)) vertices-in-entrance))
      (when (or (> (apply #'max vertex-values) (/ (y-of-cube place-target-cube) 2))
                (< (apply #'min vertex-values) (/ (y-of-cube place-target-cube) -2)))
        (pushback :y prot-axes))
      ;; TODO: re-enable this by fixing bin size
      ;;;; z is depth, so check if object can be inserted by pushing
      ;;(setq vertex-values (mapcar #'(lambda (x) (aref x 2))
      ;;                            (send attached-obj-local :vertices)))
      ;;(when (> (- (apply #'max vertex-values) (apply #'min vertex-values))
      ;;         (z-of-cube place-target-cube))
      ;;  (pushback :z prot-axes))
      prot-axes))
  (:verify-place-motion (arm place-target-cube ik-results
                             &key (collision-check-index (- (length ik-results) 1)))
    (let ((place-avs (copy-object ik-results)) prot-axes)
      (dotimes (i (length ik-results))
        (when (null (elt ik-results i))
          (ros::ros-error "[:verify-place-motion] IK No.~a in place motion fails" i)
          (return-from :verify-place-motion nil))
        (when (and (= i 0)
                   (null (send *ri* :angle-vector-make-trajectory (elt ik-results i)
                               :ctype (send *ri* :get-arm-controller arm :head t)
                               :move-arm arm :total-time 4000 :start-offset-time 0.01
                               :clear-velocities t :use-torso nil)))
          (ros::ros-error "[:verify-place-motion] planning motion No.~a fails" i)
          (return-from :verify-place-motion nil))
        (when (= i collision-check-index)
          (send *baxter* :angle-vector (elt ik-results i))
          (setq prot-axes
                (send self :attached-obj-protruded-axes arm place-target-cube))
          (when prot-axes
            (ros::ros-error
              "[:verify-place-motion] with robot pose No.~a, obj will collide to wall in ~a axis"
              i prot-axes)
            (return-from :verify-place-motion nil))))
      place-avs)))

(defun baxtergv6_apc2016::apc-init (&key (ctype :default-controller) (moveit nil))
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
      (setq *ti* (instance baxtergv6_apc2016::apc-interface :init)))
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send *ri* :calib-grasp :arms)))
