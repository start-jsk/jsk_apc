#!/usr/bin/env roseus

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter-interface.l")


(defclass in_bin_data_collection
  :super propertied-object
  :slots (arm_)
  )
(defmethod in_bin_data_collection
  (:init
    ()
    (ros::ros-info "[~a] Initializing" (ros::get-name))
    (jsk_2016_01_baxter_apc::baxter-init)
    )
  (:get-arm-for-bin
    (bin)
    (if (find bin (list :a :d :g :j)) :larm :rarm)
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
            (ros::service-call
              (format nil "data_collection_server_~a/save_request" (arm-to-str arm_))
              req)
            :success))
        (ros::ros-warn "[~a] Failed to save data. Retrying.." (ros::get-name))
        (unix::sleep 1)
        )
      )
    )
  (:initialize
    ()
    (send *ri* :angle-vector (send *baxter* :fold-pose-back arm_) 10000)
    (send *ri* :recognize-bin-boxes)
    (send *ri* :wait-interpolation)
    (send self :wait-for-user-input)
    )
  (:mainloop
    ()
    ;; :b is not ik solvable, so we skip it.
    (dolist (bin (list :a :c :d :e :f :g :h :i :j :k :l))
      (setq arm_ (send self :get-arm-for-bin bin))
      (ros::ros-info "[~a] Moving ~a to bin ~a" (ros::get-name) arm_ (symbol2str bin))
      (ros::set-param
        (format nil "/tf_bbox_to_mask_~a_hand/target_bin_name" (arm-to-str arm_))
        (symbol2str bin)
        )
      (send *ri* :move-arm-body->bin-overlook-pose arm_ bin :scale 3.0)
      (send *ri* :wait-interpolation)
      (unix::sleep 3)
      (send self :send-save-request)
      (let (avs)
        (pushback (send *baxter* :avoid-shelf-pose arm_ bin) avs)
        (pushback (send *baxter* :fold-to-keep-object arm_) avs)
        (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 3.0)
        (send *ri* :wait-interpolation)
        )
      )
    )
  )


;; __main__
(ros::roseus "in_bin_data_collection_main" :anonymous nil)
(setq *collector* (instance in_bin_data_collection :init))
(send *collector* :initialize)
(send *collector* :mainloop)
(exit)
