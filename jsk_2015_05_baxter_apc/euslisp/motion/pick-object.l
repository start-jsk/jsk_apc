(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/check-if-grabbed.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

(defun insert-arm-to-pick-avs (arm)
  (let (avs)
    (pushback (send *baxter* arm :move-end-pos #f(0 0 30) :world :revert-if-fail nil) avs)
    (pushback (send *baxter* arm :move-end-pos #f(200 0 0) :world :revert-if-fail nil) avs)
    avs))

(defun try-to-pick-object (arm pattern)
  (let (avs)
    (pushback (send *baxter* :angle-vector) avs)
    (case pattern
      (:just-down
        (send *baxter* arm :move-end-pos #f(0 0 -70) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation)
        )
      (:left-and-down
        (send *baxter* arm :move-end-pos #f(0 50 0) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 0 -70) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation))
      (:right-and-down
        (send *baxter* arm :move-end-pos #f(0 -50 0) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 0 -70) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation))
      (:rotate-and-right
        (rotate-wrist arm -90)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 -40 0) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation))
      (:rotate-and-left
        (rotate-wrist arm 90)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 40 0) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
        (send *ri* :wait-interpolation))
      )
    (send *irtviewer* :draw-objects)
    (unix::sleep 3)
    (if (check-if-grabbed arm)
      (progn
        (setq avs (list (send *baxter* arm :move-end-pos #f(0 0 30) :world :revert-if-fail nil :rotation-axis :z)))
        (print avs)
        )
      (progn ;; go back to original position
        (nreverse avs)
        (send *baxter* :angle-vector (car (last avs)))))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 8.0)
    (send *ri* :wait-interpolation)
    ))

(defun pick-object (arm)
  (let (avs (patterns '(:just-down :left-and-down :right-and-down :rotate-and-left :rotate-and-right)))
    ;; start grasp
    (ros::ros-info (format nil "[pick_object] start grasp"))
    (send *ri* :start-grasp arm)

    (pushback (send *baxter* :angle-vector) avs)

    ;; insert arm to target bin
    (setq avs (append avs (insert-arm-to-pick-avs arm)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs)
    (send *ri* :wait-interpolation)

    ;; try to pick object while not grabbed
    (while
      (and (not (check-if-grabbed arm)) patterns)
      (setq pattern (pop patterns))
      (try-to-pick-object arm pattern)
      (unless *simulator-p* (speak-en  "Fail to catch the target" :google t))
      )

    ;; take arm out of bin
    (setq avs (list (send *baxter* arm :move-end-pos #f(-100 0 0) :world :revert-if-fail nil :rotation-axis nil)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 8.0)
    (send *ri* :wait-interpolation)
    ))
