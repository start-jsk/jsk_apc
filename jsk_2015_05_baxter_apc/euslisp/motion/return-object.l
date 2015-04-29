(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/motion/robot-motion-common.l")

;; this is function version of object-picking's release
;; this function must call from release-object
(defun return-object-avs (l/r)
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

(defun return-object (l/r)
  ;; l/r validataion
  (if (not
       (or (eq l/r :larm) (eq l/r :rarm)))
      (return-from pick-object "invalid l/r. l/r must be :larm or :rarm"))

  (setq avs (return-object-avs l/r))
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5)
  (send *ri* :wait-interpolation)

  ;; stop grasp(cleaner OFF)
  (ros::ros-info (format nil "[releasing_object] cleaner OFF"))
  (send *ri* :stop-grasp)
  (unix::sleep 5)

  ;; return to original pose
  ;; (setq avs (append avs (reverse (cdr avs))))
  (setq avs (reverse (cdr avs)))
  (send *baxter* :angle-vector (car (last avs)))
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector-sequence avs :fast :default-controller 0 :scale 2.5)
  (send *ri* :wait-interpolation)
  )