;; -*- mode: lisp;-*-

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

(defun str-to-symbol (str) (warn "str-to-symbol is deprecate. use str-to-key.~%"))
(defun symbol-to-str (_symbol) (warn "symbol-to-str is deprecate. use key-to-str.~%"))

(defun str-to-key (str)
  (intern (string-upcase str) "KEYWORD"))

(defun key-to-str (_symbol)
  (string-downcase (symbol-string _symbol)))
