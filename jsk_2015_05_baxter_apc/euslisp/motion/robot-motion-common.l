(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(defun fold-pose-up (arm)
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(-7.80029 -1.12061 -82.9248 141.438 116.477 -6.48193 8.10791))
        (av-r #f(7.80029 -1.12061 82.9248 141.438 -116.477 -6.48193 -8.10791)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (midpoint 0.5 av-init (send *baxter* :angle-vector)) (send *baxter* :angle-vector))))

(defun fold-pose-mid (&optional (arm :arms))
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(48.8672 -10.437 -80.6616 148.645 80.9033 1.38428 18.8745))
        (av-r #f(-48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745)))
    (case arm
      (:arms (progn (send *baxter* :larm :angle-vector av-l) (send *baxter* :rarm :angle-vector av-r)))
      (:larm (send *baxter* :larm :angle-vector av-l))
      (:rarm (send *baxter* :rarm :angle-vector av-r)))
    (list (midpoint 0.5 av-init (send *baxter* :angle-vector)) (send *baxter* :angle-vector))))

(defun fold-pose-low (arm)
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(-7.27295 52.6465 -59.9414 91.582 123.574 13.3374 159.675))
        (av-r #f(7.27295 52.6465 59.9414 91.582 -123.574 13.3374 -159.675)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (midpoint 0.5 av-init (send *baxter* :angle-vector)) (send *baxter* :angle-vector))))

(defun fold-pose-back (arm)
  (let ((av-init (send *baxter* :angle-vector))
        (av-l #f(97.4707 -2.39502 -94.5483 134.67 91.4062 8.70117 0))
        (av-r #f(-97.4707 -2.39502 94.5483 134.67 -91.4062 8.70117 0)))
    (case arm
      (:rarm (send *baxter* arm :angle-vector av-r))
      (:larm (send *baxter* arm :angle-vector av-l)))
    (list (midpoint 0.5 av-init (send *baxter* :angle-vector)) (send *baxter* :angle-vector))))

(defun untuck-pose (arm)
  (let ((av-init (send *baxter* :angle-vector)) av-arm)
    (send *baxter* :untuck-pose)
    (setq av-arm (send *baxter* arm :angle-vector))
    (send *baxter* :angle-vector av-init)     ; back to initial pose
    (send *baxter* arm :angle-vector av-arm)  ; only move one arm
    (list (send *baxter* :angle-vector))))

(defun rotate-wrist (arm angle &key (angle-method :relative))
  (let (wrist-index)
    (case arm
      (:larm (setq wrist-index 7))
      (:rarm (setq wrist-index 14)))
    (setq av (send *baxter* :angle-vector))
    (setf (elt av wrist-index) (+ (elt av wrist-index) angle))
    (send *baxter* :angle-vector av)))

(defun look-at-other-side (arm)
  (let (av)
    (setq av (send *baxter* :angle-vector))
    (case arm
      (:larm (setf (aref av 0) -90))
      (:rarm (setf (aref av 0) 90)))
    (send *baxter* :angle-vector av)))

(defmethod baxter-robot
  (:fold-pose
    ()
    (let (av)
      (fold-pose-back :larm)
      (fold-pose-back :rarm)
      (setq av (send *baxter* :angle-vector))
      (setf (aref av 0) 0)
      (send *baxter* :angle-vector av))))

(defmethod pod-lowres-object
  ; local coordinates of each target bin
  ; these are the entrances
  (:a () (send self :transform-vector #f(-280 1660 400)))
  (:b () (send self :transform-vector #f(   0 1660 400)))
  (:c () (send self :transform-vector #f( 280 1660 400)))
  (:d () (send self :transform-vector #f(-280 1430 400)))
  (:e () (send self :transform-vector #f(   0 1430 400)))
  (:f () (send self :transform-vector #f( 280 1430 400)))
  (:g () (send self :transform-vector #f(-280 1200 400)))
  (:h () (send self :transform-vector #f(   0 1200 400)))
  (:i () (send self :transform-vector #f( 280 1200 400)))
  (:j () (send self :transform-vector #f(-280  940 400)))
  (:k () (send self :transform-vector #f(   0  940 400)))
  (:l () (send self :transform-vector #f( 280  940 400)))
  (:bin-list () '(:a :b :c :d :e :f :g :h :i :j :k :l)))
