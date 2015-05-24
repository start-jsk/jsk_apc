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

;; (setq arm :larm bin :k)
;; (progn (load "motion/pick-object.l ")
;;        (dolist (av (insert-to-pick-object-avs arm bin))
;;          (send *irtviewer* :draw-objects) (send *ri* :angle-vector av 3000)
;;          (send *ri* :wait-interpolation))
;;        (try-to-pick-object arm bin :rotate-and-left 30)
;;        (send *ri* :stop-grasp))

(defun try-to-pick-object (arm bin pattern depth
                               &key
                               (call-check-if-grabbed-function t)
                               (call-see-if-grabbed-function nil)
                               (grasp-depth-z  95) ;; move z down to grasp
                               (check-depth-z  10) ;; move z up to check grabbed
                               (grasp-side-z   80) ;; move z down to side grasp
                               (grasp-side-z-down   50) ;; move z down to side (grasp after rotate)
                               (grasp-z-offset 30) ;; z for returning to original pose
                               (debug-mode nil)
                               )
  (let* ((grasp-left-most-dist 60)
         (grasp-right-most-dist -60)
         (grasp-left-dist 50)
         (grasp-right-dist -50)
         (grasp-middle-left-dist 20)
         (grasp-middle-right-dist -20)
         (grasp-limit-left-side 50)
         (grasp-limit-right-side -50)
         (grasp-half-limit-left-side (/ grasp-limit-left-side 2))
         (grasp-half-limit-right-side (/ grasp-limit-right-side 2))
         (depth-threshold 90)
         )
    ;; load robot-test.l and run following command to get this information
    ;; (check-pick-offset :rarm '(:c :f :i :l) '((:middle . 30)))
    ;; (check-pick-offset :larm '(:a :d :g :j :b :e :h :k) '((:middle . 30)))
    (setq grasp-depth-z (min grasp-depth-z
                             (or
                              (case arm
                                (:larm
                                 (case bin (:a 95) (:d 95) (:g 95) (:j 95) (:b 90) (:e 90) (:h 90) (:k 85))
                                 )
                                (:rarm
                                 (case bin (:c 90) (:f 80) (:i 115) (:l 130))
                                 ))
                              grasp-depth-z)))

    (if (and (> depth depth-threshold)
             (or (eq bin :a) (eq bin :b) (eq bin :c)))
        (setq grasp-side-z 70))

    (move-end-pos-with-interpolation arm :x depth :time 1000)

    ;; move left or right
    (case pattern
      (:left-most
       (move-end-pos-with-interpolation arm :y grasp-left-most-dist :time 1000)
       )
      (:right-most
       (move-end-pos-with-interpolation arm :y grasp-right-most-dist :time 1000)
       )
      (:left
       (move-end-pos-with-interpolation arm :y grasp-left-dist :time 1000)
       )
      (:right
       (move-end-pos-with-interpolation arm :y grasp-right-dist :time 1000)
       )
      (:middle
       ;; none
       )
      (:middle-left
       (move-end-pos-with-interpolation arm :y grasp-middle-left-dist :time 1000)
       )
      (:middle-right
       (move-end-pos-with-interpolation arm :y grasp-middle-right-dist :time 1000)
       )
      (:rotate-and-left
       (rotate-wrist arm 90)
       (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
       (send *ri* :wait-interpolation)
       (move-end-pos-with-interpolation arm :z (- grasp-side-z))
       )
      (:rotate-and-right
       (rotate-wrist arm -90)
       (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
       (send *ri* :wait-interpolation)
       (move-end-pos-with-interpolation arm :z (- grasp-side-z))
       )
      )

    ;; make vacuume ON
    (send *ri* :start-grasp arm)
    (start-image-time-diff arm)

    ;; grabbing
    (ros::ros-info "try to grasp with pattern ~A, depth ~A (grasp-depth-z ~A check-depth-z ~A debug-mdoe ~A)" pattern depth grasp-depth-z check-depth-z debug-mode)
    (case pattern
      ((:left-most :right-most :left :right :middle :middle-left :middle-right)
       (move-end-rot-with-interpolation arm :y 2)
       (move-end-pos-with-interpolation arm :z (- grasp-depth-z))
       )
      (:rotate-and-left
       (move-end-rot-with-interpolation arm :x -4)
       (move-end-pos-with-interpolation arm :y grasp-limit-left-side :z (- grasp-side-z-down) :time 1000)
       )
      (:rotate-and-right
       (move-end-rot-with-interpolation arm :x 4)
       (move-end-pos-with-interpolation arm :y grasp-limit-right-side :z (- grasp-side-z-down) :time 1000))
      )

    (unix::sleep 1)

    ;; for check grabbed, seperate gripper from wall.
    (case pattern
      ((:left-most :right-most :left :right :middle :middle-left :middle-right)
       (move-end-pos-with-interpolation arm :z grasp-depth-z :time 1000) ;; return to original height
       )
      (:rotate-and-left
       (move-end-pos-with-interpolation arm :z (- grasp-side-z grasp-side-z-down) :time 1000)
       (move-end-pos-with-interpolation arm :y (- grasp-half-limit-left-side) :time 1000 :force t)
       )
      (:rotate-and-right
       (move-end-pos-with-interpolation arm :z (- grasp-side-z grasp-side-z-down) :time 1000)
       (move-end-pos-with-interpolation arm :y (- grasp-half-limit-right-side) :time 1000 :force t)
       )
      ) ;; case pattern

    (when call-see-if-grabbed-function
      (if (and (not *simulator-p*) (see-if-grabbed arm))
          (progn
            (ros::ros-info "finish because see if grabbed")
            (insert-to-pick-object-avs arm bin)
            (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
            (send *irtviewer* :draw-objects)
            (return-from try-to-pick-object t)
            )
        (progn
          (ros::ros-info "continue because see if grabbed")
          (insert-to-pick-object-avs arm bin)
          (return-from try-to-pick-object nil)))) ;; if use see-if-grabbed

    (ros::ros-info "call (check-if-grabbed)")
    (unless (or debug-mode (check-if-grabbed arm))
      (ros::ros-info "go back to original position")
      ;; make vacuume OFF
      (send *ri* :stop-grasp arm)

      ;; go back to original position if can't grabbed
      (case pattern
        ((:left-most :right-most :left :right :middle :middle-left :middle-right)
         ;; (move-end-pos-with-interpolation arm :z (- grasp-depth-z check-depth-z) :time 1000) ;; for check grabbed, seperate gripper.
         )
        (:rotate-and-right)
        (move-end-pos-with-interpolation arm :y 40 :z grasp-side-z-down :time 1000)
        (:rotate-and-left
         (move-end-pos-with-interpolation arm :y -40 :z grasp-side-z-down :time 1000)
         ))
      (insert-to-pick-object-avs arm bin)
      (send *irtviewer* :draw-objects)
      (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
      (send *ri* :wait-interpolation)

      (if
          (and
           (not *simulator-p*)
           call-see-if-grabbed-function
           (see-if-grabbed arm)
           )
          (progn
            (ros::ros-info "finish because see if grabbed")
            (return-from try-to-pick-object t)
            )
        (return-from try-to-pick-object nil)))
    ;; patterns after grabbed
    (ros::ros-info "take arm from target bin")

    ;; return to original pose.
    (case pattern
      ((:left-most :right-most :left :right :middle :middle-left :middle-right)
       ;; 50mm above then insert-to-pic
       ;; (move-end-pos-with-interpolation arm :z (+ 50 (- grasp-depth-z check-depth-z)) :revert-if-fail nil :rotation-axis nil :time 4000)
       )
      ((:rotate-and-left :rotate-and-right)
       (move-end-pos-with-interpolation arm :z grasp-side-z-down :revert-if-fail nil :rotation-axis nil :time 4000)
       )
      )

    (case pattern
      (:left-most
       (move-end-pos-with-interpolation arm :y (- grasp-left-most-dist) :time 1500)
       )
      (:right-most
       (move-end-pos-with-interpolation arm :y (- grasp-right-most-dist) :time 1500)
       )
      (:left
       (move-end-pos-with-interpolation arm :y (- grasp-left-dist) :time 1500)
       )
      (:right
       (move-end-pos-with-interpolation arm :y (- grasp-right-dist) :time 1500)
       )
      (:middle
       ;; none
       )
      (:middle-left
       (move-end-pos-with-interpolation arm :y (- grasp-middle-left-dist) :time 1500)
       )
      (:middle-right
       (move-end-pos-with-interpolation arm :y (- grasp-middle-right-dist) :time 1500)
       )
      (:rotate-and-left
       (rotate-wrist arm -45)
       (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
       (send *ri* :wait-interpolation)
       (move-end-pos-with-interpolation arm :z grasp-z-offset :time 1000)
       (move-end-pos-with-interpolation arm
                                        :y (- grasp-half-limit-left-side grasp-limit-left-side)
                                        :time 1000
                                        :force t)
       )
      (:rotate-and-right
       (rotate-wrist arm 45)
       (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
       (send *ri* :wait-interpolation)
       (move-end-pos-with-interpolation arm :z grasp-z-offset :time 1000)
       (move-end-pos-with-interpolation arm
                                        :y (- grasp-half-limit-left-side grasp-limit-right-side)
                                        :time 1000
                                        :force t)
       )
      )

    (case pattern
      ((:left-most :right-most :left :right :middle :middle-left :middle-right)
       (move-end-rot-with-interpolation arm :y -2)
       )
      )
    t))

(defun select-target-pattern-from-labels (target-labels)
  (let ((len (length target-labels))
        patterns
        (short-range 30)
        (middle-range 80)
        (long-range 150)
        )
    (labels ((neighbor (index)
                       (let ((cnt 0)))
                       (dolist (c '(1 -1 6 -6))
                         (if (and (<= 0 (+ c index)) (< c 19)
                                  (and (= c 1) (not (or (= index 6) (= index 12) (= index 18))))
                                  (and (= c -1) (not (or (= index 1) (= index 7) (= index 13))))
                                  (find (* 1.0 (+ c index)) target-labels)
                                  )
                             (setq cnt (1+ cnt))
                         )
                         )))
      (dotimes (i len)
        (let* ((label (elt target-labels i))
              (c))
          (setq c (neighbor label))
          (case label
            (1.0
             (case c
               (0 (pushback (cons :left-most long-range) patterns))
               ((1 2 3) (pushback (cons :rotate-and-left middle-range) patterns)
                (pushback (cons :left-most middle-range) patterns)
                )))
            (2.0
             (case c
               ((1 2 3) (pushback (cons :left long-range) patterns))))
            (3.0
             (case c
               ((1 2 3)
                (pushback (cons :middle-left long-range) patterns)
                (pushback (cons :middle middle-range) patterns))))
            (4.0
             (case c
               ((1 2 3)
                (pushback (cons :middle long-range) patterns)))
             )
            (5.0
             (case c
               ((1 2 3) (pushback (cons :right long-range) patterns))))
            (6.0
             (case c
               (0 (pushback (cons :right-most long-range) patterns))
               ((1 2 3)
                (pushback (cons :rotate-and-right middle-range) patterns)
                (pushback (cons :right-most middle-range) patterns))))
            (7.0
             (pushback (cons :rotate-and-left middle-range) patterns)
             (pushback (cons :left-most middle-range) patterns)
             )
            (8.0
             (pushback (cons :left middle-rnage) patterns)
             (pushback (cons :middle-left middle-range) patterns)
             )
            (9.0
             (pushback (cons :middle-left middle-range) patterns)
             (pushback (cons :middle middle-range) patterns)
             )
            (10.0
             (pushback (cons :middle middle-range) patterns)
             (pushback (cons :middle-right middle-range) patterns)
             )
            (11.0
             (pushback (cons :right middle-rnage) patterns)
             (pushback (cons :middle-right middle-range) patterns)
             )
            (12.0
             (pushback (cons :rotate-and-right middle-range) patterns)
             (pushback (cons :right-most middle-range) patterns)
             )
            (13.0
             (case c
               ((0 1 2 3)
                (pushback (cons :rotate-and-left short-range) patterns)
                (pushback (cons :left-most short-range) patterns))
               )
             )
            (14.0
             (pushback (cons :left short-range) patterns)
             (pusbback (cons :left-most short-range) patterns)
             )
            (15.0
             (pushback (cons :middle-left short-range) patterns)
             (pushback (cons :left short-range) patterns)
             (pushback (cons :middle short-range) patterns)
             )
            (16.0
             (pushback (cons :middle-right short-range) patterns)
             (pushback (cons :middle short-range) patterns)
             (pushback (cons :left short-range) patterns)
             )
            (17.0
             (pushback (cons :left short-range) patterns)
             (pushback (cons :left-most short-range) patterns)
             (pushback (cons :middle-left short-range) patterns)
             )
            (18.0
             (case c
               ((0 1 2 3)
                (pushback (cons :rotate-and-right short-range) patterns)
                (pushback (cons :right-most short-range) patterns)))
             )
            )
          )
        )
      patterns)
    ))

(defun delete-duplicate-patterns (patterns &key (overlap 1))
  (let ((pat-map (make-hash-table :test #'equal))
        result-patterns)
    (dolist (pat patterns)
      (cond ((null (gethash pat pat-map))
             (setf (gethash pat pat-map) 1)
             (pushback pat result-patterns)
             )
            ((< (gethash pat pat-map) overlap)
             (setf (gethash pat pat-map) (1+ (gethash pat pat-map)))
             (pushback pat result-patterns)
             )
            (t
             (format t "deleted pattern ~A~%" pat)
             )
          )
      )
    result-patterns))

(defun pick-object (arm bin &key (to-see-if-grabbed nil))
  (let* (av avs target-labels
            (patterns ;; pair of (pattern . depth)
             '((:middle . 30) (:middle . 80) (:middle . 150)
                       (:rotate-and-left . 30 ) (:left-most . 30)
                       (:rotate-and-right . 30) (:right-most . 30) ;;
                       (:rotate-and-left . 80) (:left-most . 80)
                       (:rotate-and-right . 80) (:right-most . 80) ;;
                       (:rotate-and-left . 150) (:left-most . 150)
                       (:rotate-and-right . 150) (:right-most . 150) ;;
                       (:left . 80) (:right . 80) (:left . 30) (:right . 30) (:left . 150) (:right . 150)
                       (:middle-left . 80) (:middle-right . 80) (:middle-left . 30) (:middle-right . 30) (:middle-left . 150) (:middle-right . 150)
                       ))
            )
    (send *baxter* arm :inverse-kinematics
          (make-cascoords :pos (v+ (send *pod* bin) #f(90 0 0)))
          :revert-if-fail nil
          :rotation-axis :z)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
    (send *irtviewer* :draw-objects)

    (setq target-labels (detect-object-pos-in-bin arm))
    (setq patterns (append (select-target-pattern-from-labels target-labels) patterns))
    (setq patterns (delete-duplicate-patterns patterns :overlap 2))

    ;; insert arm to target bin
    (dolist (av (insert-to-pick-object-avs arm bin))
      (send *irtviewer* :draw-objects)
      (send *ri* :angle-vector av 3000)
      (send *ri* :wait-interpolation))

    ;; detect object from arm's camera
    ;; store image to compare it with one after trying to pick
    (start-image-time-diff arm)
    ;; make vacuum on
    (send *ri* :start-grasp arm)
    ;; try to pick object while not grabbed
    (ros::ros-info "try to pick object ~A ~A" arm bin)
    (while
        (and patterns
             (not (try-to-pick-object arm bin (caar patterns) (cdar patterns):call-see-if-grabbed-function to-see-if-grabbed)))
      (pop patterns)
      (unless *simulator-p* (speak-en  "Fail to catch the target" :google t)))
    ;; take arm out of bin
    (ros::ros-info "take arm out of bin ~A ~A" arm bin)
    (setq avs (list (send *baxter* arm :move-end-pos #f(-250 0 0) :world :revert-if-fail nil :rotation-axis :z)))
    (pushback (avoid-shelf-pose-for-bin arm bin) avs)
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 8.0)
    (send *ri* :wait-interpolation)
    ))
