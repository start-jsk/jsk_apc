#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")

(defun model2real (&key (wait? t) (scale 16.0))
  (send *ri* :angle-vector-sequence (list (send *baxter* :angle-vector)) :fast nil 0 :scale scale)
  (when wait?
    (send *ri* :wait-interpolation)
    )
  )

(defun real2model (&key (wait? nil))
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *irtviewer* :redraw)
  )


(defun setup ()
  (jsk_2016_01_baxter_apc::baxter-init :ctype :larm-controller)
  (objects (list *baxter*))

  (send *ri* :gripper-servo-on :larm)

  (send *baxter* :fold-pose-back)
  (model2real)
  (send *ri* :wait-interpolation)
  )

(defun main ()
  (while (send *ri* :wait-for-user-input-to-start :larm)
    ;; larm-overlook-table
    (let (avs)
      (send *baxter* :larm :angle-vector
        #f(97.4927 -29.5312 -1.53809 142.537 81.167 -57.7441 27.7075 0.0)
        )
      (model2real :scale 8.0)

      (send *baxter* :larm :angle-vector
        #f(76.9043 -35.5078 15.1611 67.7197 131.902 -77.4536 -5.64697 0.0)
        )
      (pushback (send *baxter* :angle-vector) avs)

      (send *baxter* :larm :angle-vector
        #f(37.4414 -45.6152 27.356 47.2412 154.072 -86.7261 8.78906 0.0)
        )
      (pushback (send *baxter* :angle-vector) avs)

      (send *baxter* :larm :angle-vector
        #f(2.83447 -60.6885 46.4722 63.7207 156.401 -85.1221 33.9258 0.0)
        )
      (pushback (send *baxter* :angle-vector) avs)

      (send *baxter* :larm :angle-vector
        #f(-45.1538 -40.3857 51.8994 67.1924 174.968 -76.7065 60.1392 0.0)
        )
      (pushback (send *baxter* :angle-vector) avs)

      (ros::service-call "/kinfu/reset" (instance std_srvs::EmptyRequest :init))

      (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 8.0)
      (send *ri* :wait-interpolation)

      (ros::service-call "/kinfu/save_mesh" (instance std_srvs::EmptyRequest :init))
      (send *ri* :angle-vector-sequence (reverse avs) :fast nil 0 :scale 5.0)
      (send *ri* :wait-interpolation-smooth 2000)
    )

    (send *baxter* :fold-pose-back)
    (model2real :scale 8.0)
    )
  )
