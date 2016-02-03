;; vim: set ft=lisp:


(defmacro pushback (el lst)
  `(if (null ,lst)
       (setf ,lst (list ,el))
     (nconc ,lst (list ,el))))


(defun m->mm (m) (* 1000 m))


(defun argmax (fvec)
  (let (ret val maxv)
    (dotimes (i (length fvec))
      (setq val (aref fvec i))
      (cond ((= i 0) (setq maxv val ret i))
            ((> val maxv) (setq maxv val ret i))
            (t nil)))
    ret))


(defun vec-list-max (vec-list &key axis)
  (let (vec max-vec)
    (dotimes (i (length vec-list))
      (setq vec (elt vec-list i))
      (cond
        ((= i 0) (setq max-vec vec))
        ((< (elt max-vec axis) (elt vec axis)) (setq max-vec vec))
        (t nil)
        )
      )
    max-vec))


(defun str2symbol (str)
  (intern (string-upcase str) "KEYWORD"))


(defun symbol2str (_symbol)
  (string-downcase (symbol-string _symbol)))


(defun ros::advertise-if-yet (name data-class queue-size)
  (unless (ros::get-topic-publisher name)
    (ros::advertise name data-class queue-size)
    (unix::sleep 1)))


(defun underscore-to-space (str_)
  (let* ((str (copy-list str_)))
    (while
      (position #\_ str)
      (setf (schar str (position #\_ str)) #\ ))
    str))

(defun which-bin-region (bin)
  (let ((l-bins '(:a :d :g :j))
        (c-bins '(:b :e :h :k))
        (r-bins '(:c :f :i :l)))
    (cond
      ((member bin l-bins) :left)
      ((member bin c-bins) :center)
      ((member bin r-bins) :right)
      (t nil))))


(defun arm-to-ctype (arm)
  (case arm
    (:larm :larm-controller)
    (:rarm :rarm-controller)))


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
