#!/usr/bin/env roseus

(ros::roseus "robot_main")

(require "package://baxter_paper_filing/euslisp/lib/paper-filing-interface-roseus-resume.l")

(defun paper-filing-init (&key (ctype :default-controller) (moveit nil))
  (baxter_paper_filing::paper-filing-init :ctype ctype :moveit moveit)
  ;; Currently, moveit env is set but not used.
  ;; (when moveit
  ;;   (send *ti* :wipe-all-scene)
  ;;   (send *ti* :add-workspace-scene)
  ;;   )
  ;; initialize fingers
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pre-pose :rarm :cylindrical) 1000 :wait nil)
  (send *ri* :move-hand :larm
        (send *baxter* :hand-grasp-pre-pose :larm :cylindrical) 1000)
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pre-pose :rarm :opposed) 1000 :wait nil)
  (send *ri* :move-hand :larm
        (send *baxter* :hand-grasp-pre-pose :larm :opposed) 1000)
  (send *ri* :angle-vector (send *baxter* :arc-reset-pose))
  (send *ri* :wait-interpolation)
  (objects (list *baxter*))
  t)

(warn "~% Commands ~%")
(warn "(paper-filing-init :ctype :larm-controller :moveit t) : initialize *ti*~%")
(warn "(send *ti* :mainloop)                                 : start the mainloop~%~%")
