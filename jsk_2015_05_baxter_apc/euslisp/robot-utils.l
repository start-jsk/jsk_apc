;; -*- mode: lisp;-*-

(defun str-to-arm (str)
  (cond
    ((string= str "left") :larm)
    ((string= str "right") :rarm)
    (t nil)))

(defun arm-to-str (arm)
  (cond
    ((eq arm :larm) "left")
    ((eq arm :rarm) "right")
    (t nil)))

(defun opposite-arm (arm)
  (cond
    ((eq arm :larm) :rarm)
    ((eq arm :rarm) :larm)
    (t nil)))

(defun ir2ri (&optional (tm nil))
  (send *ri* :angle-vector (send *baxter* :angle-vector) tm)
  (send *irtviewer* :draw-objects))

(defun ri2ir ()
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *irtviewer* :draw-objects))

(defun force-to-reach-goal (&key (arm :arms) (threshold 5) (stop 10))
  (let ((times 0) arms)
    (case arm
      (:arms (setq arms '(:larm :rarm)))
      (:larm (setq arms '(:larm)))
      (:rarm (setq arms '(:rarm)))
      )
    (dolist (arm arms)
      (while
        (> (norm (v- (send *baxter* arm :angle-vector) (send *ri* :arm-potentio-vector arm))) threshold)
        (when (> times stop) (ros::ros-warn "abort to send avs: ~A" arm) (return))  ;; abort
        (ros::ros-info "difference of avs: ~A" (norm (v- (send *baxter* arm :angle-vector) (send *ri* :arm-potentio-vector arm))))
        (send *ri* :angle-vector (send *baxter* :angle-vector))
        (send *ri* :wait-interpolation)
        (incf times)
        )
      ))
  )
