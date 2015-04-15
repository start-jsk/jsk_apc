;;
;; utils
;;
(defmacro pushback (item place)
  `(setf ,place (append ,place (list ,item))))

(defun argmax (fvec)
  (let (ret)
    (dotimes (i (length fvec))
      (setq val (aref fvec i))
      (cond ((= i 0) (setq maxv val ret i))
            ((> val maxv) (setq maxv val ret i))
            (t nil)))
    ret))

(defun str2symbol (str)
  (intern (string-upcase str) "KEYWORD"))

