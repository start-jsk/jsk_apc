;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defun insert-arm-to-return-avs (arm bin)
  (let (avs)
    (send *baxter* arm :inverse-kinematics (make-cascoords :pos (v+ (send *pod* bin) #f(-150 0 50))) :revert-if-fail nil)
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
    (send *ri* :wait-interpolation)
    (send *baxter* arm :move-end-pos #f(400 0 0) :world :revert-if-fail nil :rotation-axis :z)
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
    (send *ri* :wait-interpolation)
  avs))

(defun return-object (arm bin)
  (let ((init-angle (send *baxter* :angle-vector)) avs shake-avs)
    ;; insert arm to target bin
    (ros::ros-info "[return_object] insert arm to bin")
    (setq avs (insert-arm-to-return-avs arm bin))
    (ros::ros-info "[return_object] finish inserting arm")

    ;; stop grasp(cleaner OFF)
    (ros::ros-info "[return_object] stop grasp")
    (send *ri* :stop-grasp arm)
    (ros::ros-info "[return_object] wait for object to leave from hand")
    (unix::sleep 5)
    ;; shake to have object leave from hand
    (dotimes (i 3)
      (pushback (rotate-wrist arm 10) shake-avs)
      (pushback (rotate-wrist arm -20) shake-avs)
      (pushback (rotate-wrist arm 20) shake-avs)
      (pushback (rotate-wrist arm -10) shake-avs))
    (send *ri* :angle-vector-sequence shake-avs)
    (send *ri* :wait-interpolation)
    (unix::sleep 2)
    (ros::ros-info "[return_object] finish waiting")

    ;; return to original position
    (ros::ros-info "[return_object] take arm out of bin")
    (nreverse avs)
    (pushback init-angle avs)
    (send *baxter* :angle-vector (car (last avs)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs)
    (send *ri* :wait-interpolation)
    (ros::ros-info "[return_object] finish taking arm out of bin")
  ))
