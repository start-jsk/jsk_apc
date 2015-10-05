;; vim: set ft=lisp:
;; -*- mode: lisp;-*-
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/model/pod-lowres.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/model/order-bin.l")

(setq *debug-view* nil)
(setq *simulator-p* nil)
(defun apc-init (&optional (arm :default))
  (let ((ctype
          (case arm
            (:default :default-controller)
            (:larm :larm-controller)
            (:rarm :rarm-controller))))
    (baxter-init :safe nil :type ctype)
    (send *baxter* :locate #f(0 0 950) :world)  ;; baxter in the lab
    ;; (send *baxter* :locate #f(0 0 1040) :world)  ;; baxter in the challenge
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
    (send (send *baxter* :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
    (send (send *baxter* :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
    (send (send (*ri* . robot) :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
    (send (send (*ri* . robot) :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
    (if (not (boundp '*tfb*)) (setq *tfb* (instance ros::transform-broadcaster :init)))
    ;; (send *tfb* :send-transform (send (make-coords) :translate #f(0 0 -950) :world) "base" "pod")
    (send *tfb* :send-transform (send (make-coords) :translate #f(0 0 -1040) :world) "base" "pod")
    (pod-init)
    (orderbin-init)
    (objects (list *baxter* *pod* *orderbin*))
    (setq *simulator-p* (not (send *ri* :joint-action-enable)))
    t))
