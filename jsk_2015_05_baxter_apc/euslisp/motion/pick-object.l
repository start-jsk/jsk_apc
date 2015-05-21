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

(defun try-to-pick-object (arm bin pattern depth &key (call-check-if-grabbed-function t))
  ;; move left or right
  (case pattern
    (:right-most
     (move-end-pos-with-interpolation arm :y -80 :time 1000)
     )
    (:left-most
     (move-end-pos-with-interpolation arm :y 80 :time 1000)
     )
    (:right
     (move-end-pos-with-interpolation arm :y -50 :time 1000)
     )
    (:left
     (move-end-pos-with-interpolation arm :y 50 :time 1000)
     )
    (:middle
     ;; none
     )
    (:middle-left
     (move-end-pos-with-interpolation arm :y 20 :time 1000)
     )
    (:middle-right
     (move-end-pos-with-interpolation arm :y -20 :time 1000)
     )
    (:rotate-and-right
     (rotate-wrist arm -90)
     (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
     (send *ri* :wait-interpolation)
     (move-end-pos-with-interpolation arm :y -100 :time 1000))
    (:rotate-and-left
     (rotate-wrist arm 90)
     (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
     (send *ri* :wait-interpolation)
     (move-end-pos-with-interpolation arm :y 100 :time 1000))
    )

  (move-end-pos-with-interpolation arm :x depth :time 1000)
  (case pattern
    ((:right-most :left-most :right :left :middle :middle-left :middle-right)
     (move-end-pos-with-interpolation arm :z -120)
     ))

  (send *irtviewer* :draw-objects)
  (unix::sleep 3)

  (send *ri* :angle-vector (send *baxter* arm :move-end-pos #f(0 0 30)) 1500)
  (send *ri* :wait-interpolation)

  ;; check if arm gripper unexpectedly attached to shelf or not
  (ros::ros-info "check if attached to shelf")
  (setq tmp-av (send *baxter* :angle-vector))
  (setq arm-av (send *baxter* arm :angle-vector))
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (ros::ros-info "potentio diff: ~A" (norm (v- (send *baxter* arm :angle-vector) arm-av)))
  (send *baxter* :angle-vector tmp-av)

  (ros::ros-info "check if grabbed")
  (unless call-check-if-grabbed-function ;; return without checking grabbed
    (insert-to-pick-object-avs arm bin)
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (return-from try-to-pick-object nil))
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
     (move-end-pos-with-interpolation arm :z 120 :revert-if-fail nil :rotation-axis nil :time 4000)
     ))

  (case pattern
    (:right-most
     (move-end-pos-with-interpolation arm :y 80 :time 1500)
     )
    (:left-most
     (move-end-pos-with-interpolation arm :y -80 :time 1500)
     )
    (:right
     (move-end-pos-with-interpolation arm :y 50 :time 1500)
     )
    (:left
     (move-end-pos-with-interpolation arm :y -50 :time 1500)
     )
    (:middle
     ;; none
     )
    (:middle-left
     (move-end-pos-with-interpolation arm :y -20 :time 1500)
     )
    (:middle-right
     (move-end-pos-with-interpolation arm :y 20 :time 1500)
     )
    (:rotate-and-right
     (rotate-wrist arm -90)
     (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
     (send *ri* :wait-interpolation)
     (move-end-pos-with-interpolation arm :y -100 :time 1000))
    (:rotate-and-left
     (rotate-wrist arm 90)
     (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
     (send *ri* :wait-interpolation)
     (move-end-pos-with-interpolation arm :y 100 :time 1000))
    )

  ;; rotate wrist to original
  (case pattern
    (:rotate-and-left
     (rotate-wrist arm 90))
    (:rotate-and-right
     (rotate-wrist arm -90)))
  t)

(defun pick-object (arm bin)
  (let* (av avs (patterns
                '(:right-most :right :middle-right :middle :midle-left :left :left-most :rotate-and-left :rotate-and-right
                              :right-most :right :middle-right :middle :midle-left :left :left-most :rotate-and-left :rotate-and-right
                              :right-most :right :middle-right :middle :midle-left :left :left-most :rotate-and-left :rotate-and-right))
           (depths (append (make-list 9 :initial-element 0)
                           (make-list 9 :initial-element 80)
                           (make-list 9 :initial-element 150))))
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
