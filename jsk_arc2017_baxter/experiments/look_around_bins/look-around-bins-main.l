#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(defun model2real (&key (wait? t) (scale nil))
  (send *ri* :angle-vector-sequence (list (send *baxter* :angle-vector))
        :fast nil 0
        :scale (or scale 8.0))
  (when wait?
    (send *ri* :wait-interpolation-smooth 3000)
    )
  )

(defun real2model (&key (wait? nil))
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *irtviewer* :redraw)
  )

(defun setup ()
  (jsk_arc2017_baxter::arc-init :ctype :rarm-controller)
  (objects (list *baxter*))
  (setq *arm-angles* nil)
  (setq *arm-angles-file*
    (format nil "~a/experiments/look_around_bins/~a"
            (ros::rospack-find "jsk_arc2017_baxter")
            "data/look-around-bins-rarm-angles.l" *arm-angles*)
    )
  )

(defun record ()
  (real2model)
  (pushback (send *baxter* :rarm :angle-vector) *arm-angles*)
  (dump-loadable-structure *arm-angles-file*)
  (ros::ros-info-green "Saved arm angles to: ~a" *arm-angles-file*)
  )

(defun play-step-by-step (&key (sim t) (scale))
  (dolist (aa *arm-angles*)
    (ros::ros-info-green "Playing arm-angle: ~a" aa)
    (send *baxter* :rarm :angle-vector aa)
    (send *irtviewer* :redraw)
    (if sim
      (progn (unix::sleep 1))
      (progn
        (model2real :wait? nil :scale scale)
        (send *ri* :wait-interpolation-smooth time-offset)
        )
      )
    )
  )

(defun play (&key (scale))
  (let (avs)
    (dolist (aa *arm-angles*)
      (ros::ros-info-green "Playing arm-angle: ~a" aa)
      (send *baxter* :rarm :angle-vector aa)
      (send *irtviewer* :redraw)
      (pushback (send *baxter* :angle-vector) avs)
      )
    (send *ri* :angle-vector (elt avs 0) :fast nil 0 :scale 5)
    (send *ri* :wait-interpolation)
    ;;
    (ros::service-call "/kinfu/reset" (instance std_srvs::EmptyRequest :init))
    ;;
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale (or scale 5))
    (send *ri* :wait-interpolation)
    )
  )

(defun record-main ()
  (while
    (when (y-or-n-p "Record? ") (record))
    )
  )

(defun play-main (&key (scale))
  (load *arm-angles-file*)
  (play :scale scale)
  )


(setup)
(warn "=========== Usage ==============
(play-main): Play recorded motions
(record-main): Record motions~%")
