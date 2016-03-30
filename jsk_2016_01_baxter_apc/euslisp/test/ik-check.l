#!/usr/bin/env roseus
;; *able* : ik-solvable range in either arm
;; *larm-able* : ik-solvable range only in left arm
;; *rarm-able* : ik-solvable range only in right arm
;; *disbale* : ik-unsolvable range in both arms

(load "package://jsk_2015_05_baxter_apc/euslisp/test/data/bin-cubes.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")

(defun check-init ()
  (setq *baxter* (instance jsk_2015_05_baxter_apc::baxter-robot :init))
  (setq *cubes* (get-cubes-list *bin-cubes*))
  (setq *bin-keys* (list :a :b :c :d :e :f :g :h :i :j :k :l))
  (objects (append (list *baxter*) *cubes*)))

(defun range-init ()
  (setq *able* nil)
  (setq *larm-able* nil)
  (setq *rarm-able* nil)
  (setq *disable* nil))

(defun dump-range ()
  (dump-loadable-structure "ik-range.l" *baxter* *able* *larm-able* *rarm-able* *disable*))

(defun get-cube (key)
  (gethash key *bin-cubes*))

(defun get-cubes-list (table)
  (remove nil (mapcar #'cdr (send table :list))))

(defun visualize ()
  (objects (pushback *baxter* (append *able* *disable*))))

(defun visualize-able ()
  (objects (pushback *baxter* *able*)))

(defun visualize-larm-able ()
  (objects (pushback *baxter* *larm-able*)))

(defun visualize-rarm-able ()
  (objects (pushback *baxter* *rarm-able*)))

(defun visualize-disable ()
  (objects (pushback *baxter* *disable*)))

(defun check-baxter-ik-all-bin (&optional (step-dist 100))
  (range-init)
  (dolist (key *bin-keys*)
    (format t "bin :~A~%" key)
    (setq cube (get-cube key))
    (setq dx (x-of-cube cube))
    (setq dy (y-of-cube cube))
    (setq dz (z-of-cube cube))
    (setq x (elt (send (send cube :worldcoords) :pos) 0))
    (setq y (elt (send (send cube :worldcoords) :pos) 1))
    (setq z (elt (send (send cube :worldcoords) :pos) 2))

    (do ((tmp-x (+ (- x (/ dx 2)) (/ step-dist 2)) (+ tmp-x step-dist)))
      ((> tmp-x (+ x (/ dx 2))) t)
      (do ((tmp-y (+ (- y (/ dy 2)) (/ step-dist 2)) (+ tmp-y step-dist)))
        ((> tmp-y (+ y (/ dy 2))) t)
        (do ((tmp-z (+ (- z (/ dz 2)) (/ step-dist 2)) (+ tmp-z step-dist)))
        ((> tmp-z (+ z (/ dz 2))) t)
        (setq pos (float-vector tmp-x tmp-y tmp-z))
        (format t "X: ~A~%Y: ~A~%Z: ~A~%" tmp-x tmp-y tmp-z)
        (setq larm-bool t)
        (setq rarm-bool t)
        (send *baxter* :reset-pose)
        (if (send *baxter* :larm :inverse-kinematics
                  (make-cascoords :pos pos)
                  :rotation-axis t
                  :debug-view :no-message)
          (format t "success to solve ik in left arm~%")
          (setq larm-bool nil))
        (send *baxter* :reset-pose)
        (if (send *baxter* :rarm :inverse-kinematics
                  (make-cascoords :pos pos)
                  :rotation-axis t
                  :debug-view :no-message)
          (format t "success to solve ik in right arm~%")
          (setq rarm-bool nil))
        (if (or larm-bool rarm-bool)
            (progn
              (cond
              ((and larm-bool (not rarm-bool))
                (setq larm-step-cube (make-cube step-dist step-dist step-dist :pos pos))
                (send larm-step-cube :set-color :blue)
                (pushback larm-step-cube *larm-able*))
              ((and (not larm-bool) rarm-bool)
                (setq rarm-step-cube (make-cube step-dist step-dist step-dist :pos pos))
                (send rarm-step-cube :set-color :yellow)
                (pushback rarm-step-cube *rarm-able*))
              (t nil))
              (setq step-cube (make-cube step-dist step-dist step-dist :pos pos))
              (send step-cube :set-color :green)
              (pushback step-cube *able*))
          (progn
            (setq dis-step-cube (make-cube step-dist step-dist step-dist :pos pos))
            (send dis-step-cube :set-color :red)
            (pushback dis-step-cube *disable*)
            (format t "failed to solve ik in both arm~%"))))))
  t))
