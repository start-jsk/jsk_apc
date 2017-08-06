;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")


(defmethod baxter-robot
  (:ik-prepared-poses
    ()
    (list
      :reset-pose
      :untuck-pose
      :fold-pose-upper
      :fold-pose-middle
      :fold-pose-lower
      :fold-pose-back
      :fold-to-keep-object
      )
    )
  )

(defun run (limb)
  (do-until-key
    (let ((avs)
          (tc)
          (mt (send *robot* limb :end-coords)))
      ;;
      (if (eq limb :larm)
        (setq av (send *robot* :view-hand-pose limb :g))
        (setq av (send *robot* :view-hand-pose limb :i))
        )
      (pushback av avs)
      ;;
      (setq tc (send *robot* limb :end-coords :copy-worldcoords))
      (send tc :rotate (* (random 36) (/ 2pi 36)) :x)
      (send tc :rotate (* (random 36) (/ 2pi 36)) :y)
      (send tc :rotate (* (random 36) (/ 2pi 36)) :z)
      (setq av
            (send *robot* limb :inverse-kinematics
                  tc
                  ;; fit the rotation
                  ; :use-gripper t :translation-axis :xy :thre 200 :rthre (deg2rad 60)
                  :use-gripper t :translation-axis t :thre 200 :rthre (/ 2pi 18)
                  ;; second task with null-space, fit the position
                  :additional-check #'(lambda ()
                                        (and
                                          (> 1 (norm (send mt :difference-position tc)))
                                          (> (deg2rad 1) (norm (send mt :difference-rotation tc)))
                                          ))
                  :debug-view nil
                  )
            )
      (pushback av avs)
      ;;
      (send *robot* :visualize-path avs)
      (unix::sleep 3)
      )
    )
  )


;; __main__
;; ------------------------------------------------------------------------------------------------

(setq *baxter* (instance jsk_2016_01_baxter_apc::baxter-robot :init))
(setq *robot* *baxter*)

(setq base->kinect2-torso
      (make-coords
        :pos (float-vector (* 1000 0.21) (* 1000 0.075) (* 1000 0.45))
        :rpy (float-vector -0.01 -1.57 3.14)
        )
      )

(setq base (send *robot* :base_lk :worldcoords))

(setq kinect2-torso (send *robot* :base_lk :worldcoords :copy-worlcoords))
(send kinect2-torso :transform base->kinect2-torso)

(objects
  (list
    *robot*
    base
    kinect2-torso
    )
  )

(warn "(run :rarm): random view-hand-pose~%")
