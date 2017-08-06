#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter-interface.l")

(ros::roseus "loop_ik_bin_entrance")

(setq arm :rarm)
(if (eq arm :larm)
  (setq bin-list (list :a :b :d :e :g :h :j :k))
  (setq bin-list (list :c :e :f :h :i :k :l :b))
  )

(jsk_2016_01_baxter_apc::baxter-init)
(send *ri* :gripper-servo-on arm)
(send *ri* :angle-vector
      (send *baxter* :fold-to-keep-object arm)
      5000)
(send *ri* :wait-interpolation)
(send *ri* :angle-vector
      (send *baxter* :fold-pose-back arm)
      5000)
(send *ri* :wait-interpolation)

(send *ri* :recognize-bin-boxes)

(dolist (bin bin-list)
  (ros::ros-info "[~a] Moving arm ~a to bin ~a" (ros::get-name) arm bin)
  (send *ri* :move-arm-body->bin arm bin)
  (send *ri* :wait-interpolation)
  (send *ri* :angle-vector
        (send *ri* :ik->bin-entrance arm bin
              :offset #f(0 0 0) :gripper-angle 0)
        3000)
  (send *ri* :wait-interpolation)

  (do-until-key
    (ros::ros-info "[~a] Sleeping 1 seconds" (ros::get-name))
    (unix::sleep 1)
    )

  (send *ri* :angle-vector
    (send *baxter* :avoid-shelf-pose arm bin)
    5000)
  (send *ri* :wait-interpolation)

  (send *ri* :angle-vector
        (send *baxter* :fold-to-keep-object arm)
        5000)
  (send *ri* :wait-interpolation)
  )
(exit)
