#!/usr/bin/env roseus

(ros::roseus "sanity_check_gripper_v6" :anonymous nil)

(require "package://jsk_arc2017_baxter/euslisp/lib/baxter-interface.l")

(defun sanity-check-gripper-v6
  ()
  (jsk_arc2017_baxter::baxter-init)

  ;; sanity check for fingers
  (let ((l-finger-av-init (send *ri* :get-real-finger-av :larm))
        (r-finger-av-init (send *ri* :get-real-finger-av :rarm))
        l-finger-av r-finger-av)
    (setq l-finger-av (v+ l-finger-av-init (float-vector 45 45)))
    (setq r-finger-av (v+ r-finger-av-init (float-vector 45 45)))
    (send *ri* :move-hand :larm l-finger-av 2000)
    (send *ri* :move-hand :rarm r-finger-av 2000 :wait t)

    (setq l-finger-av (v+ l-finger-av-init (float-vector -45 -45)))
    (setq r-finger-av (v+ r-finger-av-init (float-vector -45 -45)))
    (send *ri* :move-hand :larm l-finger-av 2000)
    (send *ri* :move-hand :rarm r-finger-av 2000 :wait t)

    (send *ri* :move-hand :larm l-finger-av-init 2000)
    (send *ri* :move-hand :rarm r-finger-av-init 2000 :wait t)
    )

  ;; sanity check for prismatic joint
  (let ((l-gripper-x-init (send *baxter* :larm :gripper-x :joint-angle))
        (r-gripper-x-init (send *baxter* :rarm :gripper-x :joint-angle))
        (l-gripper-p-init (send *baxter* :larm :gripper-p :joint-angle))
        (r-gripper-p-init (send *baxter* :rarm :gripper-p :joint-angle)))
    (send *baxter* :slide-gripper :larm 30 :relative t)
    (send *baxter* :slide-gripper :rarm 30 :relative t)
    (send *baxter* :rotate-gripper :larm 30 :relative t)
    (send *baxter* :rotate-gripper :rarm 30 :relative t)
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector))
    (send *ri* :wait-interpolation)

    (send *baxter* :slide-gripper :larm -60 :relative t)
    (send *baxter* :slide-gripper :rarm -60 :relative t)
    (send *baxter* :rotate-gripper :larm -60 :relative t)
    (send *baxter* :rotate-gripper :rarm -60 :relative t)
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector))
    (send *ri* :wait-interpolation)

    (send *baxter* :slide-gripper :larm l-gripper-x-init :relative nil)
    (send *baxter* :slide-gripper :rarm r-gripper-x-init :relative nil)
    (send *baxter* :rotate-gripper :larm l-gripper-p-init :relative nil)
    (send *baxter* :rotate-gripper :rarm r-gripper-p-init :relative nil)
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector))
    (send *ri* :wait-interpolation)
    )

  ;; sanity check for suction
  (dolist (arm (list :larm :rarm))
    (send *ri* :start-grasp arm)
    (unix::sleep 1)
    (send *ri* :stop-grasp arm)
    (unix::sleep 1)
    )
  )

(sanity-check-gripper-v6)
(exit)
