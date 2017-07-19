#!/usr/bin/env roseus
;; vim: set ft=lisp:

;; Init
;; ====
(ros::roseus "collect_twod_datset_main" :anonymous nil)
(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

;; Parameters
;; ==========
(setq box-type (ros::get-param "~box_type" "bin"))
(let ((side (ros::get-param "~side" "right")))
  (cond ((string= side "both") (setq *arms* '(:larm :rarm)))
        ((string= side "left") (setq *arms* '(:larm)))
        ((string= side "right") (setq *arms* '(:rarm)))
        )
  (ros::ros-info-green "Collect rawdata for creating dataset2d in [~a] boxes, using ~a hand" box-type side)
  )

;; Main
;; ====
(jsk_arc2017_baxter::arc-init :ctype :default-controller :moveit nil)
(objects (list *baxter*))
(cond ((string= box-type "bin") (send *ti* :recognize-bin-boxes))
      ((string= box-type "tote") (send *ti* :recognize-tote-boxes))
      (t (ros::ros-info-red "Invalid input rosparam ~box_type. Use \"bin\" or \"tote\"."))
      )

(defun rqt_yn_btn (&key (message ""))
  (let ((req (instance jsk_gui_msgs::YesNoRequest :init)))
    (send req :message message)
    (ros::wait-for-service "rqt_yn_btn")
    (send (ros::service-call "rqt_yn_btn" req) :yes)))

(ros::ros-info-green "Start collecting raw data in ~a" box-type)
(while
  t
  (let ((bins '(:a :b :c)) (totes '(:tote)) bins-or-totes)
    (when (rqt_yn_btn :message "Can I collect rawdata?")
      (setq bins-or-totes (cond ((string= box-type "bin") bins)
                                ((string= box-type "tote") totes)
                                ))
      (dolist (bin-or-tote bins-or-totes)
        (let ((arm (cond ((string= box-type "bin") (if (eq bin-or-tote :a) :larm :rarm))
                         ((string= box-type "tote") :rarm)
                         )))
          (when (find arm *arms*)
            (ros::set-param (format nil "/~a_hand/target_~a_name" (arm-to-str arm) box-type)
                            (symbol2str bin-or-tote))
            (ros::set-param (format nil "/~a_hand/view_frame" (arm-to-str arm)) "default")

            ;; overlook the target box
            (cond ((string= box-type "bin") (send *ti* :move-arm-body->bin-overlook-pose arm bin-or-tote))
                  ((string= box-type "tote") (send *ti* :move-arm-body->tote-overlook-pose arm))
                  )
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
  )
