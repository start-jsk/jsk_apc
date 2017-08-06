#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/baxter-interface.l")

(ros::roseus "loop_fold_and_verify_pose")

(setq arm :rarm)

(defun setup ()
  (jsk_2015_05_baxter_apc::baxter-init :ctype :default-controller)
  (send *ri* :angle-vector (send *baxter* :fold-pose-back) 3000)
  (send *ri* :wait-interpolation)
  )

(defun main ()
  (do-until-key
    (send *ri* :angle-vector
          (send *baxter* :fold-to-keep-object arm))
    (send *ri* :wait-interpolation)
    ;; fold pose back
    (send *ri* :angle-vector
      (send *baxter* :fold-pose-back arm))
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp arm)
    (unix::sleep 2)
    (send *ri* :start-grasp arm)
    (unix::sleep 2)
    ;; move arm body to head
    (send *ri* :move-arm-body->head-view-point arm)
    (send *ri* :wait-interpolation)
    ;; record image
    (unix::sleep 3)
    )
  (send *ri* :stop-grasp)
  )

(setup)
(main)
