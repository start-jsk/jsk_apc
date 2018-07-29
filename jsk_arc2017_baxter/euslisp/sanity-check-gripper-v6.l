#!/usr/bin/env roseus

(ros::roseus "sanity_check_gripper_v6" :anonymous nil)

(require "package://jsk_arc2017_baxter/euslisp/lib/baxter-interface.l")

(defun sanity-check-gripper-v6
  ()
  (setq arms nil)
  (let ((check-left (ros::get-param "~left" t))
        (check-right (ros::get-param "~right" t))
        ctype)
    (cond
      ((and check-left check-right) (setq ctype :default-controller arms (list :larm :rarm)))
      (check-left (setq ctype :larm-head-controller arms (list :larm)))
      (check-right (setq ctype :rarm-head-controller arms (list :rarm)))
      )
    (jsk_arc2017_baxter::baxter-init :type ctype)
    )

  (dolist (arm arms)
    ;; sanity check for fingers
    (let ((finger-av-init (send *ri* :get-real-finger-av arm)) finger-av)
      (setq finger-av (v+ finger-av-init (float-vector 45 45)))
      (send *ri* :move-hand arm finger-av 2000 :wait t)

      (setq finger-av (v+ finger-av-init (float-vector -45 -45)))
      (send *ri* :move-hand arm finger-av 2000 :wait t)

      (send *ri* :move-hand arm finger-av-init 2000 :wait t)
      )

    ;; sanity check for prismatic joint
    (let ((gripper-x-init (send *baxter* arm :gripper-x :joint-angle))
          (gripper-p-init (send *baxter* arm :gripper-p :joint-angle)))
      (send *baxter* :slide-gripper arm 30 :relative t)
      (send *baxter* :rotate-gripper arm 30 :relative t)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector))
      (send *ri* :wait-interpolation)

      (send *baxter* :slide-gripper arm -60 :relative t)
      (send *baxter* :rotate-gripper arm -60 :relative t)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector))
      (send *ri* :wait-interpolation)

      (send *baxter* :slide-gripper arm gripper-x-init :relative nil)
      (send *baxter* :rotate-gripper arm gripper-p-init :relative nil)
      (send *ri* :angle-vector-raw (send *baxter* :angle-vector))
      (send *ri* :wait-interpolation)
      )

    ;; sanity check for suction
    (send *ri* :start-grasp arm)
    (unix::sleep 1)
    (send *ri* :stop-grasp arm)
    (unix::sleep 1)
    )
  )

(sanity-check-gripper-v6)
(exit)
