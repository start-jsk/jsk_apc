#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")


(defun move-arm-body->bin-overlook-pose->multi-view
  (arm bin view-frame &key (gripper-angle 90))
  (let (avs offset rpy
            (offset-y -180) (pi/4 (/ pi 4.0)) (-pi/4 (- (/ pi 4.0)))
            (pi/6 (/ pi 6.0)) (pi/12 (/ pi 12.0)) (pi/15 (/ pi 15.0)))
    (setq avs (list (send *baxter* :avoid-shelf-pose arm (if (eq arm :larm) :d :f))))
    (send *baxter* :reset-pose arm)
    (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
    (setq offset
      (cond ((eq bin :a) (float-vector 200 offset-y 250))
            ((eq bin :b) (cond ((eq view-frame :1) (float-vector -150 offset-y 150))
                               ((eq view-frame :2) (float-vector -250 0 150))
                               ((eq view-frame :3) (float-vector -100 (- offset-y) 150))
                               ((eq view-frame :4) (float-vector 50 (- offset-y) 200))
                               ((eq view-frame :5) (float-vector -100 0 200))
                               ((eq view-frame :6) (float-vector -50 offset-y 200))
                               ((eq view-frame :7) (float-vector -120 -100 250))
                               ((eq view-frame :8) (float-vector -120 0 250))
                               ((eq view-frame :9) (float-vector -100 50 250))))
            ((eq bin :c) (cond ((eq view-frame :1) (float-vector 20 -100 150))
                               ((eq view-frame :2) (float-vector 20 0 150))
                               ((eq view-frame :3) (float-vector 50 100 150))
                               ((eq view-frame :4) (float-vector 100 (- offset-y) 200))
                               ((eq view-frame :5) (float-vector -100 0 200))
                               ((eq view-frame :6) (float-vector 0 offset-y 200))
                               ((eq view-frame :7) (float-vector -50 -100 250))
                               ((eq view-frame :8) (float-vector -50 0 250))
                               ((eq view-frame :9) (float-vector -50 100 250))))))
    (setq rpy
      (cond ((eq bin :b) (cond ((eq view-frame :1) (float-vector 0 (- pi/2 pi/6) -pi/4))
                               ((eq view-frame :2) (float-vector 0 (- pi/2 pi/6) 0))
                               ((eq view-frame :3) (float-vector 0 (- pi/2 pi/6) pi/4))
                               ((eq view-frame :4) (float-vector 0 pi/2 pi/2))
                               ((eq view-frame :5) (float-vector 0 pi/2 0))
                               ((eq view-frame :6) (float-vector 0 pi/2 -pi/2))
                               ((eq view-frame :7) (float-vector 0 (+ pi/2 pi/15) -pi/4))
                               ((eq view-frame :8) (float-vector 0 (+ pi/2 pi/15) 0))
                               ((eq view-frame :9) (float-vector 0 (+ pi/2 pi/15) pi/4))))
            ((eq bin :c) (cond ((eq view-frame :1) (float-vector 0 (+ pi/2 pi/6) -pi/4))
                               ((eq view-frame :2) (float-vector 0 (+ pi/2 pi/6) 0))
                               ((eq view-frame :3) (float-vector 0 (+ pi/2 pi/6) pi/4))
                               ((eq view-frame :4) (float-vector 0 pi/2 pi/2))
                               ((eq view-frame :5) (float-vector 0 pi/2 0))
                               ((eq view-frame :6) (float-vector 0 pi/2 -pi/2))
                               ((eq view-frame :7) (float-vector 0 (+ pi/2 pi/12) -pi/4))
                               ((eq view-frame :8) (float-vector 0 (+ pi/2 pi/12) 0))
                               ((eq view-frame :9) (float-vector 0 (+ pi/2 pi/12) pi/4))))))
    (pushback
      (send *ti* :ik->bin-center arm bin
            :offset offset :rpy rpy :use-gripper nil)
      avs)
    (send *ri* :angle-vector-sequence avs :fast
          (send *ri* :get-arm-controller arm) 0 :scale 5.0)))


(defun main ()
  (setq *arm* :rarm)

  (jsk_arc2017_baxter::arc-init :ctype :rarm-head-controller :moveit nil)
  ;; (send *ti* :wipe-all-scene)

  ;; rotate head to opposite side of moving arm
  (send *baxter* :head_pan :joint-angle 70)
  (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 :head-controller 0)
  (send *ri* :wait-interpolation)

  (send *ti* :recognize-bin-boxes)
  (objects (append (list *baxter*) (send *ti* :visualize-bins)))

  ;; (send *ti* :add-cardboard-scene)
  ;; (send *ti* :add-bin-scene :b)
  ;; (send *ti* :add-bin-scene :c)

  (send *baxter* :fold-pose-back *arm*)
  (send *baxter* :head_pan :joint-angle 80)
  (send *ti* :send-av)

  ;; (send *ti* :delete-bin-scene :b)
  ;; (send *ti* :delete-bin-scene :c)

  (while (send *ti* :wait-for-user-input-to-start *arm*)
    (dolist (bin (list :b :c))
      (dolist (view-frame (list :1 :2 :3 :4 :5 :6 :7 :8 :9))
        ;; (ros::set-param "/right_hand/target_bin_name" (symbol2str bin))
        (cond ((eq bin :b) (ros::set-param "/right_hand/target_bin_name" "tort_back"))
              ((eq bin :c) (ros::set-param "/right_hand/target_bin_name" "tort_front")))
        (ros::set-param "/right_hand/view_frame" (symbol2str view-frame))
        ;; (send *ti* :add-bin-scene :b)
        ;; (send *ti* :add-bin-scene :c)
        ;; (send *ti* :move-arm-body->bin-overlook-pose *arm* bin)
        (move-arm-body->bin-overlook-pose->multi-view *arm* bin view-frame)
        ;; (send *baxter* *arm* :move-end-pos #f(0 0 100) :world)
        (send *ti* :send-av)
        (send *irtviewer* :redraw)
        (send *ri* :wait-interpolation)
        ;; (send *ti* :delete-bin-scene :b)
        ;; (send *ti* :delete-bin-scene :c)

        (unix::sleep 3)
        (setq res
              (ros::service-call "/right_hand/data_collection_server/save_request"
                                 (instance std_srvs::TriggerRequest :init)))
        (ros::ros-info-green "bin: ~A, view_frame: ~A, success: ~A, message: ~A" bin view-frame (send res :success) (send res :message))

        (send *baxter* :fold-pose-back *arm*)
        (send *ti* :send-av)
        )
      )
    )
  )


(main)
