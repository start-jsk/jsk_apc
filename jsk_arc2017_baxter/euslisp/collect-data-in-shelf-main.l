#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")

(setq *arm* :rarm)

(jsk_arc2017_baxter::arc-init)
(send *baxter* :fold-pose-back *arm*)
(send *ti* :send-av)

(send *ti* :recognize-bin-boxes)
(objects (append (list *baxter*) (send *ti* :visualize-bins)))

(while (send *ti* :wait-for-user-input-to-start *arm*)
  (dolist (bin (list :b :c))
    (ros::set-param "/right_hand/target_bin_name" (symbol2str bin))
    (send *ti* :move-arm-body->bin-overlook-pose *arm* bin)
    (send *baxter* *arm* :move-end-pos #f(0 0 100) :world)
    (send *ti* :send-av)
    (send *irtviewer* :redraw)
    (send *ri* :wait-interpolation)

    (unix::sleep 1)
    (setq res
          (ros::service-call "/right_hand/data_collection_server/save_request"
                            (instance std_srvs::TriggerRequest :init)))
    (ros::ros-info-green "success: ~A, message: ~A" (send res :success) (send res :message))

    (send *baxter* :fold-pose-back *arm*)
    (send *ti* :send-av)
    )
  )
