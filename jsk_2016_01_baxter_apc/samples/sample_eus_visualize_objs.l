#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter-interface.l")


(defmethod jsk_2016_01_baxter_apc::baxter-interface
  (:graspingp (arm) t)
  )


(jsk_2016_01_baxter_apc::baxter-init)
(setq *irtviewer* (make-irtviewer))
(send *irtviewer* :change-background #f(0.8 0.8 0.8))

(setq right-gripper-bodies nil)
(setq all-bodies (send *baxter* :bodies))
(setq keep (list 23 24 25 30 31))
(dotimes (i (length all-bodies))
  (when (find i keep)
    (pushback (elt all-bodies i) right-gripper-bodies)
    )
  )
(objects right-gripper-bodies)

(setq bin-box-msg
      (one-shot-subscribe "/bbox_array_to_bbox/output" jsk_recognition_msgs::BoundingBox))
(sethash :f (*ri* . _bin-boxes) bin-box-msg)
(setq obj-boxes-msg
      (one-shot-subscribe "/right_sib_cpi_decomposer/boxes"
                          jsk_recognition_msgs::BoundingBoxArray))
(sethash :f (*ri* . _objects-sib-boxes) (send obj-boxes-msg :boxes))
(setq obj-coms-msg
      (one-shot-subscribe "/right_sib_cpi_decomposer/centroid_pose_array"
                          geometry_msgs::PoseArray))
(sethash :f (*ri* . _objects-sib-coords)
         (mapcar #'(lambda (obj-pose)
                     (send *ri* :tf-pose->coords
                           (send obj-coms-msg :header :frame_id) obj-pose))
                 (send obj-coms-msg :poses)))

(setq bin-viz (send *ri* :visualize-bins))
(setq obj-viz (send *ri* :visualize-objects))
(objects (append right-gripper-bodies bin-viz obj-viz))

(warn ";; Please hit ctrl-c to stop~%")
(send *ri* :try-to-pick-object :rarm :f)
(send *irtviewer* :redraw)
(while (ros::ok)
       (send *irtviewer* :draw-objects)
       (x::window-main-one)
       (unix::usleep 100000)
       )
