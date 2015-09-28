;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_baxter_apc/euslisp/model/pod-lowres.l")
(load "package://jsk_2014_baxter_apc/euslisp/model/order-bin.l")


;; parameters
(setq *debug-view* nil)


(defun apc-init (&optional (arm :default) (baxter-pos #f(0 0 950)))
  (let (ctype)

    (setq ctype
          (case arm
            (:default :default-controller)
            (:larm :larm-controller)
            (:rarm :rarm-controller)))

    ;; initialize baxter
    (baxter-init :safe nil :type ctype)
    ;; customize baxter
    (send (send *baxter* :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
    (send (send *baxter* :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
    (send (send (*ri* . robot) :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
    (send (send (*ri* . robot) :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))

    ;; robot location
    (send *baxter* :locate baxter-pos :world)

    ;; apply robot current state
    (send *baxter* :angle-vector (send *ri* :state :potentio-vector))

    ;; set tf publisher
    (if (not (boundp '*tfb*)) (setq *tfb* (instance ros::transform-broadcaster :init)))
    (send *tfb* :send-transform (send (make-coords) :translate baxter-pos :world) "base" "pod")

    ;; initialize other objects
    (pod-init)
    (orderbin-init)

    ;; visualize objects
    (objects (list *baxter* *pod* *orderbin*))

    (setq *simulator-p* (send *ri* :simulation-modep))
    ))
