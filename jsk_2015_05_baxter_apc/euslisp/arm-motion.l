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
  (let ((av-l #f(-7.80029 -1.12061 -82.9248 141.438 116.477 -6.48193 8.10791))
        (av-r #f(7.80029 -1.12061 82.9248 141.438 -116.477 -6.48193 -8.10791)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (send *baxter* :angle-vector))))
(defun fold-pose-mid (&optional (arm :arms))
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(48.8672 -10.437 -80.6616 148.645 80.9033 1.38428 18.8745))
        (av-r #f(-48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745)) midp)
    (case arm
      (:arms (progn (send *baxter* :larm :angle-vector av-l) (send *baxter* :rarm :angle-vector av-r)))
      (:larm (send *baxter* :larm :angle-vector av-l))
      (:rarm (send *baxter* :rarm :angle-vector av-r)))
    (setq midp (midpoint 0.5 av-init (send *baxter* :angle-vector)))
    (list midp (send *baxter* :angle-vector))))
(defun fold-pose-low (arm)
  (let ((av-l #f(-7.27295 52.6465 -59.9414 91.582 123.574 13.3374 159.675))
        (av-r #f(7.27295 52.6465 59.9414 91.582 -123.574 13.3374 -159.675)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (send *baxter* :angle-vector))))
(defun fold-pose-back (arm)
  (let ((av-l #f(97.4707 -2.39502 -94.5483 134.67 91.4062 8.70117 0))
        (av-r #f(-97.4707 -2.39502 94.5483 134.67 -91.4062 8.70117 0)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (send *baxter* :angle-vector))))
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

(defun rotate-wrist (angles l/r &optional (angle 90))
  (let ((wrist-index 0))
    (case l/r
      (:larm
       (setq wrist-index 7))
      (:rarm
       (setq wrist-index 14)))
    (setf (elt angles wrist-index) (+ (elt angles wrist-index) angle))
    angles))

;; this is function version of object-picking
(defun pick-object-avs (l/r)
  (let (revavs avs)
    ;; picking pose
    (pushback (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
    (pushback (send *baxter* l/r :move-end-pos #f(0 0 30) :world :revert-if-fail nil) avs)

    (let* ((p 0.2)
           (point (midpoint p #f(0 0 0) #f(260 0 0))))
      (dotimes (i (/ 1 p))
        (pushback (send *baxter* l/r :move-end-pos point :world :revert-if-fail nil) avs)))

    ;; random motion
    (let* ((random-y 40)
           (random-move-points (list (float-vector 0 random-y 0) (float-vector 0 0 0) (float-vector 0 (- random-y) 0)))
           (range (length random-move-points))
           (sign (if (= (random 2) 1) 1 -1))
           (rotate-angle (* sign 90)))
      (if (= (random 2) 1)
          (progn
            (pushback (send *baxter* l/r :move-end-pos (elt random-move-points (random range)) :world :revert-if-fail nil) avs)
            (pushback (send *baxter* l/r :move-end-pos #f(0 0 -75) :world :revert-if-fail nil) avs)
            (setq revavs (reverse (cdr avs)))
            ;; rotate wrist some angle
            (dotimes (i 3)
              (pushback (rotate-wrist (send *baxter* :angle-vector) l/r (+ (random 7) -3)) avs))
            ;; return to original pose
            (setq avs (append avs revavs)))
        (progn
          (setq revavs (reverse (cdr avs)))
          (pushback (rotate-wrist (send *baxter* :angle-vector) l/r (* sign 90)) avs)
          (pushback (rotate-wrist (send *baxter* l/r :move-end-pos (float-vector 0 (* sign 80) 0) :world :revert-if-fail nil) l/r rotate-angle) avs)
          (dolist (av revavs)
            (setf av (rotate-wrist av l/r rotate-angle)))
          (setq avs (append avs revavs)))
          ))
    avs))

(defun pick-object (l/r)
  ;; l/r validataion
  (let (avs)
    (if (not
        (or (eq l/r :larm) (eq l/r :rarm)))
        (return-from pick-object "invalid l/r. l/r must be :larm or :rarm"))
    ;; send angle-vector-sequence to *ri*
    (setq avs (pick-object-avs l/r))
    (setq avs (append avs (fold-pose-up l/r)))
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 5.)
    (send *ri* :wait-interpolation)))

;; this is function version of object-picking's release
;; this function must call from release-object
(defun release-object-first (l/r)
  ;; releasing pose
  (setq avs nil)
  (pushback (send *baxter* :angle-vector (send *ri* :state :potentio-vector)) avs)
  (pushback (send *baxter* l/r :move-end-pos #f(0 0 30) :world :revert-if-fail nil) avs)
  (let* ((p 0.2)
        (point (midpoint p #f(0 0 0) #f(300 0 0))))
    (dotimes (i (/ 1 p))
             (pushback (send *baxter* l/r :move-end-pos point :world :revert-if-fail nil) avs)))

  ;;(pushback (send *baxter* l/r :move-end-pos (float-vector 0 (elt random-y (random (length random-y))) 0) :world :revert-if-fail nil) avs)
  ;;(push (send *baxter* l/r :move-end-pos (float-vector 0 (elt step-list counter) 0) :world :revert-if-fail nil) avs)

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

  ;; go grasp(cleaner ON)
  (ros::ros-info (format nil "[releasing_picking] cleaner OFF"))
  (send *ri* :stop-grasp)

  ;; return to original pose
  ;; (setq avs (append avs (reverse (cdr avs))))
  (setq avs (reverse (cdr avs)))
  (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5)
  (send *ri* :wait-interpolation)
  )

;; debug
(defun f (avs)
  (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5))
