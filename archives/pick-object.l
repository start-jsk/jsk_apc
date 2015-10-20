;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/motion/robot-motion-common.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-recognition.l")

(defun insert-to-pick-object-avs (arm bin)
  (let (avs)
    (pushback
     (send *baxter* arm :inverse-kinematics
           (make-cascoords :pos (v+ (send *pod* bin) #f(-90 0 50)))
           :revert-if-fail nil
           :rotation-axis :z) avs)
    (pushback
     (send *baxter* arm :move-end-pos #f(200 0 0) :world
           :revert-if-fail nil) avs)
    avs))

(defun insert-to-pick-object (arm bin)
  (send *baxter* arm :inverse-kinematics
        (make-cascoords :pos (v+ (send *pod* bin) #f(-90 0 50)))
        :revert-if-fail nil
        :rotation-axis :z)
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  (send *baxter* arm :move-end-pos #f(50 0 0) :world
        :revert-if-fail nil)
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)  
  (send *ri* :wait-interpolation)
  (send *baxter* arm :move-end-pos #f(150 0 0) :world
        :revert-if-fail nil)
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  )

;; (setq arm :larm bin :k)
;; (progn (require "motion/pick-object.l ")
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
                               (grasp-limit-left-side 50)
                               (grasp-limit-right-side -50)
                               (debug-mode nil)
                               (bin-objects nil)
                               (adjust-offset nil)
                               )
  (let* ((grasp-left-most-dist 60)
        (grasp-right-most-dist -60)
        (grasp-left-dist 50)
        (grasp-right-dist -50)
        (grasp-middle-left-dist 20)
        (grasp-middle-right-dist -20)
        (grasp-half-limit-left-side (* grasp-limit-left-side 0.9))
        (grasp-half-limit-right-side (* grasp-limit-right-side 0.9))
        )
    ;; TODO: remove below comments
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
       (send *baxter* :rotate-wrist arm 90)
       (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
       (send *ri* :wait-interpolation)
       (move-end-pos-with-interpolation arm :z (- grasp-side-z))
       )
      (:rotate-and-right
       (send *baxter* :rotate-wrist arm -90)
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

    ;; for examine offset, return from
    (if adjust-offset
        (progn
          (send *ri* :stop-grasp arm)
          (insert-to-pick-object-avs arm bin)
          (send *irtviewer* :draw-objects)
          (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
          (if (check-if-grabbed arm pattern bin-objects)
              (return-from try-to-pick-object nil)
            (return-from try-to-pick-object t)
            )
          )
      )

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
    (unless (or debug-mode (check-if-grabbed arm pattern bin-objects))
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

      (return-from try-to-pick-object nil)
      ) ;; check-if grabbed

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
       (send *baxter* :rotate-wrist arm -45)
       (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
       (send *ri* :wait-interpolation)
       (move-end-pos-with-interpolation arm :z grasp-z-offset :time 1000)
       (move-end-pos-with-interpolation arm
                                        :y (- grasp-half-limit-left-side grasp-limit-left-side)
                                        :time 1000
                                        :force t)
       )
      (:rotate-and-right
       (send *baxter* :rotate-wrist arm 45)
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
                       (let ((counter 0))
                         (dolist (c '(1 -1 4 -4))
                           (if (and (<= 0 (+ c index)) (< c 13)
                                    (not (and (= c 1) (or (= index 4) (= index 8) (= index 12))))
                                    (not (and (= c -1) (or (= index 1) (= index 5) (= index 9))))
                                    (not (null (find (+ c index) target-labels)))
                                    )
                               (setq counter (1+ counter))
                             )
                           )
                         counter)))
      (dotimes (i len)
        (let* ((label (elt target-labels i))
               (cnt))
          (setq cnt (neighbor label))
          ;; (format t "label = ~A, cnt = ~A~%" label cnt)
          (case label
            (1
             (case cnt
               (0
                (pushback (cons :left-most long-range) patterns)
                )
               ((1 2 3) (pushback (cons :rotate-and-left middle-range) patterns)
                (pushback (cons :left-most middle-range) patterns)
                )))
            (2
             (case cnt
               ((1 2 3)
                (pushback (cons :left long-range) patterns)
                (pushback (cons :middle-left long-range) patterns)
                )))
            (3
             (case cnt
               ((1 2 3)
                (pushback (cons :middle-right long-range) patterns)
                (pushback (cons :right long-range) patterns))))
            (4
             (case cnt
               (0
                (pushback (cons :rotate-and-right long-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :rotate-and-right long-range) patterns)
                (pushback (cons :right-most long-range) patterns)
                ))
               )
            (5
             (case cnt
               ((0)
                (pushback (cons :rotate-and-left middle-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :rotate-and-left middle-range) patterns)
                (pushback (cons :left-most long-range) patterns)))
             )
            (6
             (case cnt
               (0
                (pushback (cons :left middle-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :middle-left middle-range) patterns)
                (pushback (cons :left middle-range) patterns))))
            (7
             (case cnt
               (0
                (pushback (cons :right middle-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :middle-right middle-range) patterns)
                (pushback (cons :right middle-range) patterns)
                )
               )
             )
            (8
             (case cnt
               (0
                (pushback (cons :right-most middle-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :rotate-and-right middle-range) patterns)
                (pushback (cons :right-most middle-range) patterns)
                )
               )
             )
            (9
             (case cnt
               (0
                (pushback (cons :left-most short-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :rotate-and-left short-range) patterns)
                (pushback (cons :left-most short-range) patterns)
                )))
            (10
             (case cnt
               (0
                (pushback (cons :left short-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :middle-right short-range) patterns)
                (pushback (cons :middle short-range) patterns))))
            (11
             (case cnt
               (0
                (pushback (cons :right short-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :middle-right short-range) patterns)
                (pushback (cons :middle short-range) patterns)
                ))
               )
            (12
             (case cnt
               (0
                (pushback (cons :rotate-and-right short-range) patterns)
                )
               ((1 2 3)
                (pushback (cons :rotate-and-right short-range) patterns)
                (pushback (cons :right-most short-range) patterns)
                ))
             )
            )
          )
        )
      )
    patterns))

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

(defun delete-arbitrary-patterns (patterns arbitrary-patterns)
  (let ((result-patterns patterns))
    (dolist (a-pat arbitrary-patterns)
      (setq result-patterns (remove-if #'(lambda (x) (equal x a-pat)) result-patterns))
      )
    result-patterns))

(defun pick-object (arm bin &key (to-see-if-grabbed nil) (n-tried nil) (bin-objects nil))
  (let* (av avs target-labels bounding-box-hint
            (short-range 30)
            (middle-range 80)
            (long-range 150)
            (patterns ;; pair of (pattern . depth)
             '((:middle . short-range) (:middle . middle-range) (:middle . long-range)
                       (:rotate-and-left . short-range ) (:left-most . short-range)
                       (:rotate-and-right . short-range) (:right-most . short-range) ;;
                       (:rotate-and-left . middle-range) (:left-most . middle-range)
                       (:rotate-and-right . middle-range) (:right-most . middle-range) ;;
                       (:rotate-and-left . long-range) (:left-most . long-range)
                       (:rotate-and-right . long-range) (:right-most . long-range) ;;
                       (:left . middle-range) (:right . middle-range) (:left . short-range) (:right . short-range) (:left . long-range) (:right . long-range)
                       (:middle-left . middle-range) (:middle-right . middle-range) (:middle-left . short-range) (:middle-right . short-range) (:middle-left . long-range) (:middle-right . long-range)
                       ))
            )
;    (send *baxter* arm :inverse-kinematics
;          (make-cascoords :pos (v+ (send *pod* bin) #f(90 0 0)))
;          :revert-if-fail nil
;          :rotation-axis :z)
;    (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
;    (send *irtviewer* :draw-objects)

    (if n-tried
        (setq patterns (delete-arbitrary-patterns patterns '((:middle . long-range)))))

    (setq target-labels (detect-object-pos-in-bin arm))
    (setq patterns (append (select-target-pattern-from-labels target-labels) patterns))
    ;; (setq patterns (append (select-target-pattern-from-labels #i(1 2 3 4 5 6 7 8 9 10 11 12))))
    (setq patterns (delete-duplicate-patterns patterns :overlap 1))

    ;; (setq patterns (delete-arbitrary-patterns patterns '((:middle . 30) (:middle . 80) (:middle . 150))))
    (setq bounding-box-hint (gethash bin *bounding-box-array-in-bin*))
    (setq max-volume 0)
    (setq target-bounding-box nil)

    (when bounding-box-hint
      (dolist (bbox (send bounding-box-hint :boxes))
        (if (< max-volume (* (send bbox :dimensions :x) (send bbox :dimensions :y) (send bbox :dimensions :z)))
            (progn
              (setq max-volumet (* (send bbox :dimensions :x) (send bbox :dimensions :y) (send bbox :dimensions :z)))
              (setq target-bounding-box bbox)
              )
          )
        )
      (when target-bounding-box
        (let ((depth-offset 1.02) (y-pos ( - (* (send target-bounding-box :pose :position :y) 1000) (case bin ((:a :d :g :j) 309) ((:b :e :h :k) 0) ((:c :f :i :l) -309)))) y-label)
          (cond ((> y-pos 60) (setq y-label :left-most))
                ((> y-pos 50) (setq y-label :left))
                ((> y-pos 20) (setq y-label :middle-left))
                ((> y-pos -20) (setq y-label :middle))
                ((> y-pos -50) (setq y-label :middle-right))
                ((> y-pos -60) (setq y-label :right))
                (t (setq y-label :right-most))
                )
          (setq patterns (append (list (cons y-label ( * (- (send target-bounding-box :pose :position :x) depth-offset) 1000))) patterns))
          (ros::ros-info "Boundging Box Hints Suggest ~A" (cons y-label ( * (- (send target-bounding-box :pose :position :x) depth-offset) 1000)))
          (setq relative-pos-x ( - (* (send target-bounding-box :pose :position :x) 1000) (elt (send *pod* bin) 0)))
          (setq relative-pos-y ( - (* (send target-bounding-box :pose :position :y) 1000) (elt (send *pod* bin) 1)))
          (setq relative-pos-z ( - (* (send target-bounding-box :pose :position :z) 1000) (elt (send *pod* bin) 2)))
          (ros::ros-info " Bounding Box Absolute Pos x,y,z = ~A, ~A, ~A" (send target-bounding-box :pose :position :x) (send target-bounding-box :pose :position :y) (send target-bounding-box :pose :position :z))
          (ros::ros-info " Bounding Box Relative Pos x,y,z = ~A, ~A, ~A" relative-pos-x relative-pos-y relative-pos-z)
          (setq patterns (delete-duplicate-patterns patterns :overlap 1))
          ))
      )

    ;; (format t "patterns = ~A~%" patterns)

    ;; insert arm to target bin
    (insert-to-pick-object arm bin)

    ;; detect object from arm's camera
    ;; store image to compare it with one after trying to pick
    (start-image-time-diff arm)
    ;; make vacuum on
    (send *ri* :start-grasp arm)
    ;; try to pick object while not grabbed
    (ros::ros-info "try to pick object ~A ~A ~A" arm bin patterns)
    (while
        (and patterns
             (not (try-to-pick-object arm bin (caar patterns) (eval (cdar patterns))
                                      :call-see-if-grabbed-function to-see-if-grabbed
                                      :bin-objects bin-objects)))
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
