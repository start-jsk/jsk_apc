#!/usr/bin/env roseus
;; vim: set ft=lisp:

;; Init
;; ====
(ros::roseus "collect_twod_datset_main" :anonymous nil)
(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

;; Main
;; ====
(jsk_arc2017_baxter::arc-init :ctype :default-controller :moveit nil)
(objects (list *baxter*))
(send *ti* :recognize-bin-boxes)

(defun rqt_yn_btn (&key (message ""))
  (let ((req (instance jsk_gui_msgs::YesNoRequest :init)))
    (send req :message message)
    (ros::wait-for-service "rqt_yn_btn")
    (send (ros::service-call "rqt_yn_btn" req) :yes)))

(ros::ros-info-green "Start collecting raw data in bins")
(while
  t
  (let ((bins '(:a :b :c)))
    (when (rqt_yn_btn :message "Can I collect rawdata from bins?")
      (dolist (bin bins)
        (let ((arm (if (eq bin :a) :larm :rarm)))
          (ros::set-param (format nil "/~a_hand/target_bin_name" (arm-to-str arm))
                          (symbol2str bin))
          (ros::set-param (format nil "/~a_hand/view_frame" (arm-to-str arm)) "default")

          ;; overlook the target bin
          (send *ti* :move-arm-body->bin-overlook-pose arm bin)
          (send *ri* :wait-interpolation)

          ;; request saving raw data
          (unix::sleep 3)
          (ros::service-call
            (format nil "~a_hand/data_collection_server/save_request" (arm-to-str arm))
            (instance std_srvs::TriggerRequest :init))

          ;; reset-pose
          (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
          (send *ri* :angle-vector-sequence avs :fast
                (send *ri* :get-arm-controller arm) 0 :scale 3.0)
          (send *ri* :wait-interpolation)
          )
        )
      )
    )
  )
