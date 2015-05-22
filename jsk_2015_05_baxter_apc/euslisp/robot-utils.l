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
