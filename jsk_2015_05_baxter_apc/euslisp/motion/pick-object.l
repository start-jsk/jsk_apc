;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/move-to-target-bin.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-recognition.l")

(defun insert-to-pick-object-avs (arm bin)
  (let (avs)
  (move-to-target-bin-av arm bin)
  (pushback
    (send *baxter* arm :inverse-kinematics
          (make-cascoords :pos (v+ (send *pod* bin) #f(-90 0 40)))
          :revert-if-fail nil
          :rotation-axis :z) avs)
  (pushback
    (send *baxter* arm :move-end-pos #f(200 0 0) :world
          :revert-if-fail nil) avs)
  avs))

(defun try-to-pick-object (arm bin pattern)
  (let (avs)
    (pushback (send *baxter* :angle-vector) avs)
    (case pattern
      (:just-down
        (send *baxter* arm :move-end-pos #f(0 0 -90) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)
        )
      (:left-and-down
        (send *baxter* arm :move-end-pos #f(0 70 0) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 0 -90) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation))
      (:right-and-down
        (send *baxter* arm :move-end-pos #f(0 -70 0) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 0 -90) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation))
      (:rotate-and-right
        (rotate-wrist arm -90)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 0 -80) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 -50 0) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation))
      (:rotate-and-left
        (rotate-wrist arm 90)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 0 -80) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 50 0) :world :revert-if-fail nil :rotation-axis :z)
        (pushback (send *baxter* :angle-vector) avs)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)))
    (send *irtviewer* :draw-objects)
    (unix::sleep 3)
    (send *ri* :angle-vector (send *baxter* arm :move-end-pos #f(0 0 30)) 1500)
    (send *ri* :wait-interpolation)
    (ros::ros-info "check if grabbed")
    (unless (check-if-grabbed arm)
      (unless (if *simulator-p* nil (see-if-grabbed arm))
        (ros::ros-info "go back to original position")
        ;; go back to original position if can't grabbed
        (insert-to-pick-object-avs arm bin)
        (send *irtviewer* :draw-objects)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 2000)
        (send *ri* :wait-interpolation)
        (return-from try-to-pick-object nil)))
    ;; patterns after grabbed
    (ros::ros-info "take arm from target bin")
    (case pattern
      ;; rotate wrist to take standing object out of bin
      (:just-down
        (send *baxter* arm :move-end-pos #f(0 0 120) :world :revert-if-fail nil :rotation-axis nil)
        (send *irtviewer* :draw-objects)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
        (send *ri* :wait-interpolation))
      (:left-and-down
        (send *baxter* arm :move-end-pos #f(0 0 120) :world :revert-if-fail nil :rotation-axis nil)
        (send *irtviewer* :draw-objects)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 -40 0) :world)
        (send *irtviewer* :draw-objects)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation)
        )
      (:right-and-down
        (send *baxter* arm :move-end-pos #f(0 0 120) :world :revert-if-fail nil :rotation-axis nil)
        (send *irtviewer* :draw-objects)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
        (send *ri* :wait-interpolation)
        (send *baxter* arm :move-end-pos #f(0 40 0) :world)
        (send *irtviewer* :draw-objects)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 1500)
        (send *ri* :wait-interpolation))
      (:rotate-and-left
        (rotate-wrist arm -45)
        (send *baxter* arm :move-end-pos #f(0 -30 30) :world :revert-if-fail nil :rotation-axis nil)
        (send *irtviewer* :draw-objects)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 2000)
        (send *ri* :wait-interpolation))
      (:rotate-and-right
        (rotate-wrist arm 45)
        (send *baxter* arm :move-end-pos #f(0 30 30) :world :revert-if-fail nil :rotation-axis nil)
        (send *irtviewer* :draw-objects)
        (send *ri* :angle-vector (send *baxter* :angle-vector) 2000)
        (send *ri* :wait-interpolation)))
    t))

(defun pick-object (arm bin)
  (let (avs (patterns '(:just-down :left-and-down :right-and-down :rotate-and-left :rotate-and-right)))
    ;; insert arm to target bin
    (send *ri* :angle-vector-sequence (insert-to-pick-object-avs arm bin) :fast :default-controller 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    ;; store image to compare it with one after trying to pick
    (start-image-time-diff arm)
    ;; make vacuum on
    (send *ri* :start-grasp arm)
    ;; try to pick object while not grabbed
    (while
      (and patterns
           (not (try-to-pick-object arm bin (pop patterns))))
      (unless *simulator-p* (speak-en  "Fail to catch the target" :google t)))
    ;; take arm out of bin
    (setq avs (list (send *baxter* arm :move-end-pos #f(-250 0 0) :world :revert-if-fail nil :rotation-axis :z)))
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 8.0)
    (send *ri* :wait-interpolation)
    ))

