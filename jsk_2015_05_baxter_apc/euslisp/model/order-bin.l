;;
;; Heavy-Duty Stack and Nest Containers - Red, 24 x 15 x 8"
;;

(defun make-orderbin ()
  (let (b e)
    (setq b (make-cube 609.6 381 203.2))
    (send b :locate #f(0 0 101.6))
    (setq e (make-cube 533.4 353.6 198.12))
    (send e :locate #f(0 0 99.06))
    (send e :locate #f(0 0 5.08 ))
    (setq b (body- b e))
    (send b :set-color :red)
    b))

(defun orderbin-init ()
  (when (not (boundp '*orderbin*))
    (setq *orderbin* (make-orderbin))
    (send *orderbin* :locate #f(600 0 700) :world)
    (send *orderbin* :rotate -pi/2 :z :world)
    ))
