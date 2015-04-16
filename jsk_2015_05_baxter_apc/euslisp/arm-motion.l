;;
;;
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")

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

(defun fold-pose-mid (&optional (arm :arms))
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(37.3096 -2.30713 -74.0698 146.667 99.5361 -3.66943 6.48193))
        (av-r #f(-48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745)) midp)
    (case arm
      (:arms (progn (send *baxter* :larm :angle-vector av-l) (send *baxter* :rarm :angle-vector av-r)))
      (:larm (send *baxter* :larm :angle-vector av-l))
      (:rarm (send *baxter* :rarm :angle-vector av-r)))
    (setq midp (midpoint 0.5 av-init (send *baxter* :angle-vector)))
    (list midp (send *baxter* :angle-vector))))

(defun fold-pose-back (arm)
  (let ((av-l #f(97.4991 11.2061 -90.5933 125.574 -45.7251 -1.2085 92.7686))
        (av-r #f(-85.6274 12.4146 84.9683 149.941 -120.125 -2.68066 -6.02051)))
    (case arm
      (:rarm (send *baxter* :larm :angle-vector av-l))
      (:larm (send *baxter* :rarm :angle-vector av-r)))
    (list (send *baxter* :angle-vector))))

(defun fold-pose-up (arm)
  (let ((av-l #f(-24.1919 1.07666 -90.9009 131.572 -58.4912 -0.812988 54.7559))
        (av-r #f(7.80029 -1.12061 82.9248 141.438 -116.477 -6.48193 -8.10791)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (send *baxter* :angle-vector))))

(defun fold-pose-low (arm)
  (let ((av-l #f(-4.37256 34.9146 -40.4736 103.14 -59.9194 8.4375 91.4941))
        (av-r #f(7.27295 52.6465 59.9414 91.582 -123.574 13.3374 -159.675)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (send *baxter* :angle-vector))))

(defun fold-pose (arm bin)
  (let (avs)
    (setq avs (append avs (fold-pose-mid)))
    (setq avs (append avs (fold-pose-back arm)))
    (cond ((find bin '(:a :b :c :d :f)) (setq avs (append avs (fold-pose-up arm))))
          ((find bin '(:e :g :h :i)) (setq avs (append avs (fold-pose-mid arm))))
          ((find bin '(:j :k :l)) (setq avs (append avs (fold-pose-low arm))))
          (t nil))
    avs))

(defun move-to-bin (arm bin)
  (let (c (region (instance target-pick-region :init)))
    (setq c (send *pod* :transform-vector (send region bin)))
    (send *baxter* arm :inverse-kinematics (make-cascoords :pos c) :rotation-axis t)
    (list (send *baxter* :angle-vector))))

(defun move-arm-to-target-bin (arm bin)
  (let (avs)
    (setq avs (append avs (fold-pose arm bin)))
    (setq avs (append avs (move-to-bin arm bin)))
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5)
    (send *ri* :wait-interpolation)
    avs))

(defun verification-pose (arm)
  (let ((av-l #f(1.56006 -10.6348 -130.254 83.584 72.29 -15.4028 157.192))
        (av-r #f(-22.0825 15.1392 144.448 111.006 -25.0488 2.54883 24.895)))
    (case arm
      (:larm (send *baxter* arm :angle-vector av-l))
      (:rarm (send *baxter* arm :angle-vector av-r)))
    (list (send *baxter* :angle-vector))))

(defun move-arm-for-verification (arm)
  (let (avs)
    (setq avs (append avs (fold-pose-mid arm)))
    (setq avs (append avs (verification-pose arm)))
    (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5)
    (send *ri* :wait-interpolation)
    avs))