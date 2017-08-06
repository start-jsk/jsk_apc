;; vim: set ft=lisp:

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/baxter.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/test/data/bin-cubes.l") ;; *bin-cubes*


(defun inittest ()
  (setq *baxter* (instance jsk_2015_05_baxter_apc::baxter-robot :init))
  (setq *robot* *baxter*)
  (setq bin-cubes (remove nil (mapcar 'cdr (send *bin-cubes* :list))))
  (setq cubes nil)
  (dolist (cube-orig bin-cubes)
    (setq dim-x (elt (first (cube-orig . csg)) 1))
    (setq dim-y (elt (first (cube-orig . csg)) 2))
    (setq dim-z (elt (first (cube-orig . csg)) 3))
    (setq cube (make-cube dim-x dim-y dim-z))
    (send cube :set-color :gray 0.2)
    (send cube :transform (send cube-orig :copy-worldcoords) :world)
    (pushback cube cubes)
    )
  )


(defun runtest (limb bin)
  (let* ((cube (gethash bin *bin-cubes*))
         (path)
         (robots (list (copy-object *robot*)))
         )

    ;; goals
    (let* ((dim-x (second (first (cube . csg))))
           (goal (send cube :copy-worldcoords))
           )
      (pushback goal path)
      (setq goal (send (send (car (reverse path)) :copy-worldcoords) :translate #f(0 0 -100) :world))
      (push goal path)
      (setq goal (send (send (car (reverse path)) :copy-worldcoords) :translate (float-vector (- (/ dim-x 2)) 0 0) :world))
      (pushback goal path)
      (setq goal (send (send (car (reverse path)) :copy-worldcoords) :translate (float-vector -150 0 0) :world))
      (pushback goal path)
      )
    (send *robot* :visualize-path-planning limb path :other-objects cubes)
    )
  )

(format t "~%Usage:~%- (runtest :rarm :c): Visualize path to insert the arm into the bin.~%")
(inittest)
