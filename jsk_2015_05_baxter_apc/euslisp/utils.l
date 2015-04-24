#!/usr/bin/env roseus
;; utils.l
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

(defmacro pushback (el lst)
  `(if (null ,lst)
       (setf ,lst (list ,el))
     (nconc ,lst (list ,el))))

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
