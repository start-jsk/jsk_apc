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
