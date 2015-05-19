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
          (make-cascoords :pos (v+ (send *pod* bin) #f(-90 0 50)))
          :revert-if-fail nil
          :rotation-axis :z) avs)
  (pushback
    (send *baxter* arm :move-end-pos #f(200 0 0) :world
          :revert-if-fail nil) avs)
  avs))

(defun move-pattern-for-pick-object (arm &key (x 0) (y 0) (z 0) (time 1500) (revert-if-fail nil) (rotation-axis :z))
  (send *baxter* arm :move-end-pos (float-vector x y z) :world :revert-if-fail revert-if-fail :rotation-axis rotation-axis)
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector (send *baxter* :angle-vector) time)
  (send *ri* :wait-interpolation)
  )

(defun try-to-pick-object (arm bin pattern depth &key (rotate nil))
  ;; rotate wrist
  (case rotate
    (:left
     (rotate-wrist arm -45))
    (:right
     (rotate-wrist arm 45)))

  ;; move left or right
  (case pattern
    (:right-most
     (move-pattern-for-pick-object arm :y -100)
     )
    (:left-most
     (move-pattern-for-pick-object arm :y 100)
     )
    (:right
     (move-pattern-for-pick-object arm :y -70)
     )
    (:left
     (move-pattern-for-pick-object arm :y 70)
     )
    (:middle
     ;; none
     )
    (:middle-left
     (move-pattern-for-pick-object arm :y 20)
     )
    (:middle-right
     (move-pattern-for-pick-object arm :y -20)
     ))

  (case pattern
    ((:right-most :left-most :right :left :middle :middle-left :middle-right)
     (move-pattern-for-pick-object arm :x depth)
     (move-pattern-for-pick-object arm :z -120)
     ))

  (send *irtviewer* :draw-objects)
  (unix::sleep 3)

  (send *ri* :angle-vector (send *baxter* arm :move-end-pos #f(0 0 30)) 1500)
  (send *ri* :wait-interpolation)
  (ros::ros-info "check if grabbed")
  (unless (check-if-grabbed arm)
    (ros::ros-info "go back to original position")
    ;; go back to original position if can't grabbed
    (insert-to-pick-object-avs arm bin)
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (if (if *simulator-p* nil (see-if-grabbed arm))
        (progn (ros::ros-info "finish because see if grabbed")
               (return-from try-to-pick-object t))
      (return-from try-to-pick-object nil)))
  ;; patterns after grabbed
  (ros::ros-info "take arm from target bin")

  (case pattern
    ((:right-most :left-most :right :left :middle :middle-left :middle-right)
     (move-pattern-for-pick-object arm :z 120 :revert-if-fail nil :rotation-axis nil :time 4000)
     ))

  (case pattern
    (:right-most
     (move-pattern-for-pick-object arm :y 100 :time 1500)
     )
    (:left-most
     (move-pattern-for-pick-object arm :y -100 :time 1500)
     )
    (:right
     (move-pattern-for-pick-object arm :y 70 :time 1500)
     )
    (:left
     (move-pattern-for-pick-object arm :y -70 :time 1500)
     )
    (:middle
     ;; none
     )
    (:middle-left
     (move-pattern-for-pick-object arm :y -20 :time 1500)
     )
    (:middle-right
     (move-pattern-for-pick-object arm :y 20 :time 1500)
     ))

  ;; rotate wrist to original
  (case pattern
    (:rotate-and-left
     (rotate-wrist arm 45))
    (:rotate-and-right
     (rotate-wrist arm -45)))
  t)

(defun pick-object (arm bin)
  (let (av avs (patterns
                '(:right-most :left-most :right :left :middle :middle-left :middle-right
                              :right-most :left-most :right :left :middle :middle-left :middle-right
                              :right-most :left-most :right :left :middle :middle-left :middle-right))
           (depths '(0 0 0 0 0 0 0
                       40 40 40 40 40 40 40
                       80 80 80 80 80 80 80)))
    ;; insert arm to target bin
    (dolist (av (insert-to-pick-object-avs arm bin))
      (send *irtviewer* :draw-objects)
      (send *ri* :angle-vector av 3000)
      (send *ri* :wait-interpolation))
    ;; store image to compare it with one after trying to pick
    (start-image-time-diff arm)
    ;; make vacuum on
    (send *ri* :start-grasp arm)
    ;; try to pick object while not grabbed
    (while
      (and patterns
           (not (try-to-pick-object arm bin (pop patterns) (pop depths))))
      (unless *simulator-p* (speak-en  "Fail to catch the target" :google t)))
    ;; take arm out of bin
    (setq avs (list (send *baxter* arm :move-end-pos #f(-250 0 0) :world :revert-if-fail nil :rotation-axis :z)))
    (pushback (avoid-shelf-pose-for-bin arm bin) avs)
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 8.0)
    (send *ri* :wait-interpolation)
    ))
