(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

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

(defun untuck-pose (arm)
  (let ((av-init (send *baxter* :angle-vector)) av-arm)
    (send *baxter* :untuck-pose)
    (setq av-arm (send *baxter* arm :angle-vector))
    (send *baxter* :angle-vector av-init)     ; back to initial pose
    (send *baxter* arm :angle-vector av-arm)  ; only move one arm
    (list (send *baxter* :angle-vector))))

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

(defmethod baxter-robot
  (:fold-pose () (fold-pose-back :larm) (elt (fold-pose-back :rarm) 0)))
