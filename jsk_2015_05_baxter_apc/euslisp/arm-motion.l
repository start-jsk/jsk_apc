;;
;;
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(defun str2arm (str)
  (let (res)
    (cond ((string= str "left") (setq res :larm))
          ((string= str "right") (setq res :rarm))
          (t nil))
    res))
(defun validate-arm (arm)
  (let (res)
    (if (find arm '(:larm :rarm)) (setq res arm)
      (setq res nil))
    res))

(defun validate-bin-name (bin)
  (let ((region (instance target-pick-region :init)))
    (when (find bin (send region :list))
      (return-from validate-bin-name bin))
    (ros::ros-error "invalid bin name")
    nil))
(defun fold-pose-up (arm)
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(-7.80029 -1.12061 -82.9248 141.438 116.477 -6.48193 8.10791))
        (av-r #f(7.80029 -1.12061 82.9248 141.438 -116.477 -6.48193 -8.10791)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (midpoint 0.5 av-init (send *baxter* :angle-vector)) (send *baxter* :angle-vector))))
(defun fold-pose-mid (&optional (arm :arms))
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(48.8672 -10.437 -80.6616 148.645 80.9033 1.38428 18.8745))
        (av-r #f(-48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745)))
    (case arm
      (:arms (progn (send *baxter* :larm :angle-vector av-l) (send *baxter* :rarm :angle-vector av-r)))
      (:larm (send *baxter* :larm :angle-vector av-l))
      (:rarm (send *baxter* :rarm :angle-vector av-r)))
    (list (midpoint 0.5 av-init (send *baxter* :angle-vector)) (send *baxter* :angle-vector))))
(defun fold-pose-low (arm)
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(-7.27295 52.6465 -59.9414 91.582 123.574 13.3374 159.675))
        (av-r #f(7.27295 52.6465 59.9414 91.582 -123.574 13.3374 -159.675)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (midpoint 0.5 av-init (send *baxter* :angle-vector)) (send *baxter* :angle-vector))))
(defun fold-pose-back (arm)
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(97.4707 -2.39502 -94.5483 134.67 91.4062 8.70117 0))
        (av-r #f(-97.4707 -2.39502 94.5483 134.67 -91.4062 8.70117 0)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (midpoint 0.5 av-init (send *baxter* :angle-vector)) (send *baxter* :angle-vector))))
(defun fold-pose (arm bin)
  (let (avs)
    (setq avs (append avs (fold-pose-back (if (eq arm :larm) :rarm :larm))))  ; fold oposite arm
    (setq avs (append avs (fold-pose-mid arm)))
    (cond ((find bin '(:a :b :c)) (setq avs (append avs (fold-pose-up arm))))
          ((find bin '(:d :e :f :g :i)) (setq avs (append avs (fold-pose-mid arm))))
          ((find bin '(:h :j :k :l)) (setq avs (append avs (fold-pose-low arm))))
          (t nil))
    avs))
(defun to-target-bin (arm bin)
  (let (c (region (instance target-pick-region :init)))
    (setq c (send *pod* :transform-vector (send region bin)))
    (send *baxter* arm :inverse-kinematics (make-cascoords :pos c) :rotation-axis t)
    (list (send *baxter* :angle-vector))))
(defun verification-pose (arm)
  (let ((av-l #f(22.0825 15.1392 -144.448 111.006 25.0488 2.54883 -24.895))
        (av-r #f(-22.0825 15.1392 144.448 111.006 -25.0488 2.54883 24.895)))
    (case arm
      (:larm (send *baxter* arm :angle-vector av-l))
      (:rarm (send *baxter* arm :angle-vector av-r)))
    (list (send *baxter* :angle-vector))))
(defun untuck-pose (arm)
  (let ((av-init (send *baxter* :angle-vector)) av-arm)
    (send *baxter* :untuck-pose)
    (setq av-arm (send *baxter* arm :angle-vector))
    (send *baxter* :angle-vector av-init)     ; back to initial pose
    (send *baxter* arm :angle-vector av-arm)  ; only move one arm
    (list (send *baxter* :angle-vector))))
(defun to-order-bin (arm)
  (let (binc avs)
    (setq binc (make-cascoords :pos (send *orderbin* :transform-vector #f(0 0 300))
                               :rot #2f((-3.6e-06 0 -1) (0 1 0) (1 0 -3.6e-06))))
    (pushback (send *baxter* arm :inverse-kinematics binc :rotation-axis :z :revert-if-fail nil) avs)
    (pushback (send *baxter* arm :move-end-pos #f(0 0 -250) :world :revert-if-fail nil) avs)
    avs))

(defun move-to-order-bin (arm)
  (let (avs)
    (setq avs (append avs (fold-pose-back (if (eq arm :larm) :rarm :larm))))  ; fold oposite arm
    (setq avs (append avs (fold-pose-up arm)))
    (setq avs (append avs (untuck-pose arm)))
    (setq avs (append avs (to-order-bin arm)))
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.2)
    (send *ri* :wait-interpolation)
    avs))
(defun move-to-target-bin (arm bin)
  (let (avs)
    (setq avs (append avs (fold-pose arm bin)))
    (setq avs (append avs (to-target-bin arm bin)))
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.2)
    (send *ri* :wait-interpolation)
    avs))
(defun move-for-verification (arm)
  (let (avs)
    (setq avs (append avs (fold-pose-mid arm)))
    (setq avs (append avs (verification-pose arm)))
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.2)
    (send *ri* :wait-interpolation)
    avs))

;; this is function version of object-picking
(defun pick-object-first (l/r)
  (let (avs)
    ;; picking pose
    (pushback (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
    (pushback (send *baxter* l/r :move-end-pos #f(0 0 -20) :world :revert-if-fail nil) avs)
    (pushback (send *baxter* l/r :move-end-pos #f(260 0 0) :world :revert-if-fail nil) avs)
    avs))

(defun rotate-wrist (angles l/r &optional (angle 90))
  (let ((wrist-index 0))
    (case l/r
      (:larm
       (setq wrist-index 7))
      (:rarm
       (setq wrist-index 14)))
    (setf (elt angles wrist-index) (+ (elt angles wrist-index) angle))
    angles))

(defun same-rotate-wrist (avs l/r angles)
  (case l/r
    (:larm
     (dolist (av avs)
       (setf (elt av 7) (elt angles 7))))
    (:rarm
     (dolist (av avs)
       (setf (elt av 14) (elt angles 14)))
     ))
  avs)

(defun random-pick (l/r &optional random-num)
  ;; random motion
  (let* (avs revavs
         (random-y 50))
    (case random-num
      (0
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 0) :world :revert-if-fail nil) avs)
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 -75) :world :revert-if-fail nil) avs))
      (1
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 random-y 0) :world :revert-if-fail nil) avs)
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 -75) :world :revert-if-fail nil) avs))
      (2
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 (- random-y) 0) :world :revert-if-fail nil) avs)
       (pushback (send *baxter* l/r :move-end-pos (float-vector 0 0 -75) :world :revert-if-fail nil) avs))
      (3
       (pushback (rotate-wrist (send *baxter* :angle-vector) l/r (- 90)) avs)
       (pushback (rotate-wrist (send *baxter* l/r :move-end-pos (float-vector 0 (- 100) 0) :world :revert-if-fail nil) l/r (- 90)) avs))
      (4
       (pushback (rotate-wrist (send *baxter* :angle-vector) l/r 90) avs)
       (pushback (rotate-wrist (send *baxter* l/r :move-end-pos (float-vector 0 100 0) :world :revert-if-fail nil) l/r 90) avs))
      )
    avs))

(ros::load-ros-manifest "roseus")
(ros::roseus "grabbed")

(defparameter *grabbed* nil)
(defun grabbed-cb (msg)
  (ros::ros-info (format nil "~A" (send msg :data)))
  (setq *grabbed* (send msg :data)))
(ros::subscribe
 "gripper_grabbed/state" std_msgs::Bool #'grabbed-cb)

(defun pick-object (l/r)
  ;; l/r validataion
  (let (avs
        revavs
        (random-list (list 0 1 2 3 4)))
    (if (not
        (or (eq l/r :larm) (eq l/r :rarm)))
        (return-from pick-object "invalid l/r. l/r must be :larm or :rarm"))
    ;; go grasp(cleaner ON)
    (ros::ros-info (format nil "[picking_object] cleaner ON"))
    (send *ri* :go-grasp)

    (setq *grabbed* nil)
    (while (and (not *grabbed*) random-list)
      (ros::spin-once)
      ;; send angle-vector-sequence to *ri*
      (setq random-element (elt random-list (random (length random-list))))
      (ros::ros-info (format nil "~A~%" random-list))
      (setq avs (append (pick-object-first l/r) (random-pick l/r random-element)))
      (setq random-list (remove random-element random-list))
      (setq revavs (reverse avs))

      (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5)
      (send *ri* :wait-interpolation)
      (unix:sleep 3) ;; wait 3 seconds.

      ;; rotate same angle
      (setq revavs (same-rotate-wrist revavs l/r (car (last avs))))
      (send *ri* :angle-vector-sequence revavs :fast :default-controller 0 :scale 2.5)
      (send *ri* :wait-interpolation)
      )
    ))

;; this is function version of object-picking's release
;; this function must call from release-object
(defun release-object-first (l/r)
  ;; releasing pose
  (setq avs nil)
  (pushback (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
  (pushback (send *baxter* l/r :move-end-pos #f(0 0 30) :world :revert-if-fail nil) avs)
  (let* ((p 0.1)
        (point (midpoint p #f(0 0 0) #f(300 0 0))))
    (dotimes (i (/ 1 p))
             (pushback (send *baxter* l/r :move-end-pos point :world :revert-if-fail nil) avs)))

  (pushback (send *baxter* l/r :move-end-pos #f(0 0 -75) :world :revert-if-fail nil) avs)
  avs)

(defun release-object (l/r)
  ;; l/r validataion
  (if (not
       (or (eq l/r :larm) (eq l/r :rarm)))
      (return-from pick-object "invalid l/r. l/r must be :larm or :rarm"))

  (setq avs (release-object-first l/r))
  (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5)
  (send *ri* :wait-interpolation)

  ;; stop grasp(cleaner OFF)
  (ros::ros-info (format nil "[releasing_object] cleaner OFF"))
  (send *ri* :stop-grasp)

  ;; return to original pose
  ;; (setq avs (append avs (reverse (cdr avs))))
  (setq avs (reverse (cdr avs)))
  (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5)
  (send *ri* :wait-interpolation)
  )

;; debug
(defun f (avs)
  (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5.))
