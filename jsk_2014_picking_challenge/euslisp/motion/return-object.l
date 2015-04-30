(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

;; this is function version of object-picking's release
;; this function must call from release-object
(defun insert-arm-to-bin-avs (arm)
  (let (avs)
    (pushback (send *baxter* arm :move-end-pos #f(0 0 30) :world :revert-if-fail nil) avs)
    (pushback (send *baxter* arm :move-end-pos #f(350 0 0) :world :revert-if-fail nil) avs)
    avs))

(defun return-object (arm)
  ;; arm validataion
  (assert-arm arm)
  (let ((init-angle (send *baxter* :angle-vector)))
    ;; insert arm to target bin
    (setq avs (insert-arm-to-bin-avs arm))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 5)
    (send *ri* :wait-interpolation)

    ;; stop grasp(cleaner OFF)
    (ros::ros-info (format nil "[return_object] stop grasp"))
    (send *ri* :stop-grasp arm)
    (unix::sleep 5)

    ;; return to original position
    (nreverse avs)
    (pushback init-angle avs)
    (send *baxter* :angle-vector (car (last avs)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs)
    (send *ri* :wait-interpolation)
  ))
