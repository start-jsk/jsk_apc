;; vim: set ft=lisp:

(load "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")


(defun compute-reachable-space
  (&key (arm :rarm) (tick 10) (draw t))
  (let* (
         (av (send *robot* arm :angle-vector))
         (n-joints (length av))
         reachable-space
         reachable-space-to-draw
         coords
         i-max-angle
         i-min-angle
         j-max-angle
         j-min-angle
         )
    (dotimes (i n-joints)
      (setq i-max-angle (min 360 (send (elt (send *robot* arm) i) :joint :max-angle)))
      (setq i-min-angle (max -360 (send (elt (send *robot* arm) i) :joint :min-angle)))
      (dotimes (i-angle-index (/ (- i-max-angle i-min-angle) tick))
        (setf (elt av i) (+ i-min-angle (* tick i-angle-index)))
        (dotimes (j n-joints)
          (unless (eq i j)
            (setq j-max-angle (min 360 (send (elt (send *robot* arm) j) :joint :max-angle)))
            (setq j-min-angle (max -360 (send (elt (send *robot* arm) j) :joint :min-angle)))
            (dotimes (j-angle-index (/ (- j-max-angle j-min-angle) tick))
              (setf (elt av j) (+ j-min-angle (* tick j-angle-index)))
              (send *robot* arm :angle-vector av)
              (setq coords (send *robot* arm :end-coords :copy-worldcoords))
              (format t "av: ~a~%" av)
              (pushback coords reachable-space)
              (when draw
                (let (b)
                  (setq b (make-cube 10 10 10))
                  (send b :set-color :MediumPurple 0.5)
                  (send b :transform coords :world)
                  (pushback b reachable-space-to-draw)
                  (objects (append (list *robot*) reachable-space-to-draw))
                  )
                )
              )
            (setf (elt av j) 0)
            )
          )
        )
      (setf (elt av i) 0)
      )
    (send *robot* :reset-pose)
    reachable-space))


(defun visualize-reachable-space
  (reachable-space)
  (let (reachable-space-to-draw)
    (dolist (coords reachable-space)
      (setq b (make-cube 10 10 10))
      (send b :set-color :MediumPurple 0.5)
      (send b :transform coords :world)
      (pushback b reachable-space-to-draw)
      )
    (objects (append (list *robot*) reachable-space-to-draw))
    )
  )
