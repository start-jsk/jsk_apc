(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/check-if-grabbed.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

;; this is function version of object-picking
(defun pick-object-avs (l/r)
  ;; invalid check of l/r
  (assert-l/r l/r)
  (let (avs)
    ;; picking pose
    (pushback (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
    ;; (pushback (send *baxter* l/r :move-end-pos #f(0 0 20) :world :revert-if-fail nil) avs)
    (pushback (send *baxter* l/r :move-end-pos #f(200 0 0) :world :revert-if-fail nil) avs)
    avs))

(defun picking-pattern-avs (l/r pattern-id)
  ;; random motion
  ;; invalid check of l/r
  (assert-l/r l/r)
  (let* (avs revavs (random-y 50))
    (case pattern-id
      (0
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 0) :world :revert-if-fail nil :rotation-axis :z) avs)
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 -80) :world :revert-if-fail nil :rotation-axis :z) avs))
      (1
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 random-y 0) :world :revert-if-fail nil :rotation-axis :z) avs)
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 -80) :world :revert-if-fail nil :rotation-axis :z) avs))
      (2
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 (- random-y) 0) :world :revert-if-fail nil :rotation-axis :z) avs)
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 -80) :world :revert-if-fail nil :rotation-axis :z) avs))
      (3
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 0) :world :revert-if-fail nil :rotation-axis :z) avs)
       (pushback (rotate-wrist (send *baxter* :angle-vector) l/r (- 90)) avs)
       (pushback (rotate-wrist (send *baxter* l/r :move-end-pos (float-vector 0 (- 60) 0) :world :revert-if-fail nil :rotation-axis :z) l/r (- 90)) avs))
      (4
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 0) :world :revert-if-fail nil :rotation-axis :z) avs)
       (pushback (rotate-wrist (send *baxter* :angle-vector) l/r 90) avs)
       (pushback (rotate-wrist (send *baxter* l/r :move-end-pos (float-vector 0 60 0) :world :revert-if-fail nil :rotation-axis :z) l/r 90) avs))
      )
    avs))

(defun pick-object (l/r)
  (let (avs
        revavs
        (pattern-list (list 0 1 2 3 4))
        (init-angle (send *baxter* :angle-vector (send *ri* :state :potentio-vector))))
    ;; invalid check of l/r
    (assert-l/r l/r)

    ;; go grasp(cleaner ON)
    (ros::ros-info (format nil "[picking_object] cleaner ON"))
    (send *ri* :start-grasp)
    (send *ri* :angle-vector (send *baxter* l/r :move-end-pos #f(0 0 30) :world))
    (send *ri* :wait-interpolation)

    (setq *grabbed* nil)
    (while (and (not *grabbed*) pattern-list)
      ;; send angle-vector-sequence to *ri*
      ;;(format t "~A~%" pattern-list)
      (setq avs (append (pick-object-avs l/r) (picking-pattern-avs l/r (car pattern-list))))
      (push init-angle avs)
      (setq pattern-list (cdr pattern-list))
      (setq revavs (reverse avs))

      (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 10)
      (send *ri* :wait-interpolation)
      (unix:sleep 3) ;; wait 3 seconds.
      (ros::spin-once)  ;; check if grabbed

      ;; rotate same angle
      (setq revavs (same-rotate-wrist revavs l/r (car (last avs))))

      ;; back to original wrist angles
      (let ((av (car (last revavs))))
        (case (length pattern-list)
          (1
           (pushback (rotate-wrist av l/r 90) revavs))
          (0
           (pushback (rotate-wrist av l/r -90) revavs))))

      (send *ri* :angle-vector-sequence revavs :fast :default-controller 0 :scale 10)
      (send *ri* :wait-interpolation)
      (if *simulator-p*
          (return))
      (ros::spin-once) ;; check if grabbed
      (if (not *grabbed*)
          (speak-en  "Fail to catch the target" :google t)
        )
      )
    (send *ri* :angle-vector (send *baxter* :angle-vector init-angle))
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector (send *baxter* :move-end-pos #f(-100 0 0) :world))
    (send *ri* :wait-interpolation)
    ))
