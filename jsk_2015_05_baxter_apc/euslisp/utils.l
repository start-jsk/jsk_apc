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

(defmacro pushback (el lst)
  `(if (null ,lst)
       (setf ,lst (list ,el))
     (nconc ,lst (list ,el))))

(defun argmax (fvec)
  (let (ret val maxv)
    (dotimes (i (length fvec))
      (setq val (aref fvec i))
      (cond ((= i 0) (setq maxv val ret i))
            ((> val maxv) (setq maxv val ret i))
            (t nil)))
    ret))

(defun str-to-symbol (str)
  (intern (string-upcase str) "KEYWORD"))
(defun symbol-to-str (_symbol)
  (string-downcase (symbol-string _symbol)))

(defun assert-arm (arm)
  (if (not
       (or (eq arm :larm) (eq arm :rarm)))
      (return-from pick-object "invalid arm. arm must be :larm or :rarm")))
