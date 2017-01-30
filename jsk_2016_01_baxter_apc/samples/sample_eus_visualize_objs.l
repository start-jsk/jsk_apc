#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")


(defmethod jsk_2016_01_baxter_apc::baxter-interface
  (:graspingp (arm) t)
  )


(jsk_2016_01_baxter_apc::baxter-init)
(setq *irtviewer* (make-irtviewer))
(send *irtviewer* :change-background #f(0.8 0.8 0.8))

(setq right_gripper_bodies nil)
(setq all_bodies (send *baxter* :bodies))
(setq keep (list 23 24 25 30 31))
(dotimes (i (length all_bodies))
  (when (find i keep)
    (pushback (elt all_bodies i) right_gripper_bodies)
    )
  )
(objects right_gripper_bodies)

(setq bin_box_msg
      (one-shot-subscribe "/bbox_array_to_bbox/output" jsk_recognition_msgs::BoundingBox))
(sethash :f (*ri* . _bin-boxes) bin_box_msg)
(setq obj_boxes_msg
      (one-shot-subscribe "/right_sib_cpi_decomposer/boxes"
                          jsk_recognition_msgs::BoundingBoxArray))
(sethash :f (*ri* . _objects-sib-boxes) (send obj_boxes_msg :boxes))
(setq obj_coms_msg
      (one-shot-subscribe "/right_sib_cpi_decomposer/centroid_pose_array"
                          geometry_msgs::PoseArray))
(sethash :f (*ri* . _objects-sib-coords)
         (mapcar #'(lambda (obj-pose)
                     (send *ri* :tf-pose->coords
                           (send obj_coms_msg :header :frame_id) obj-pose))
                 (send obj_coms_msg :poses)))

(setq bin_viz (send *ri* :visualize-bins))
(setq obj_viz (send *ri* :visualize-objects))
(objects (append right_gripper_bodies bin_viz obj_viz))

(warn ";; Please hit ctrl-c to stop~%")
(send *ri* :try-to-pick-object :rarm :f)
(send *irtviewer* :redraw)
(while (ros::ok)
       (send *irtviewer* :draw-objects)
       (x::window-main-one)
       (unix::usleep 100000)
       )
