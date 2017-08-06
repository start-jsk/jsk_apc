;; vim: set ft=lisp:

(require "package://jsk_2015_05_baxter_apc/robots/baxter.l")

(unless (find-package "JSK_2015_05_BAXTER_APC")
  (make-package "JSK_2015_05_BAXTER_APC"))

(defclass jsk_2015_05_baxter_apc::baxter-robot
  :super baxter-robot
  :slots ())
(defmethod jsk_2015_05_baxter_apc::baxter-robot
  (:init
    (&rest args)
    (send-super* :init args)
    )
  (:inverse-kinematics
    (target-coords &rest args)
    (let ((av)
          (prev-av (send self :angle-vector))
          (poses (list
                   :angle-vector
                   :reset-pose
                   :reset-manip-pose
                   :fold-pose-upper
                   :fold-pose-middle
                   :fold-pose-lower
                   :fold-pose-back
                   ))
          (moving-joints (send-all (cadr (memq :link-list args)) :joint))
          (moving-av (mapcar #'(lambda (j) (send j :joint-angle)) (send self :joint-list)))
          )
      (dolist (pose poses)
        (when (not av)
          (send self pose)
          (mapcar #'(lambda (j a)
                      (if (not (memq j moving-joints))
                        (send j :joint-angle a)))
                  (send self :joint-list) moving-av)
          (setq av (send-super* :inverse-kinematics target-coords args))
          )
        )
      (when (and (null av) (or (null (memq :revert-if-fail args)) (cadr (memq :revert-if-fail args))))
          (send self :angle-vector prev-av))
      )
    (send self :angle-vector))
  (:rotate-wrist
    (arm angle)
    (let (wrist-index av-tmp)
      (setq wrist-index (if (eq arm :larm) 7 14))
      (setq av-tmp (send *baxter* :angle-vector))
      (setf (elt av-tmp wrist-index) (+ (elt av-tmp wrist-index) angle))
      (send *baxter* :angle-vector av-tmp))
    )
  (:avoid-shelf-pose
    (arm bin)
    (case bin
      (:a (send self :avoid-shelf-pose-a arm))
      (:b (send self :avoid-shelf-pose-b arm))
      (:c (send self :avoid-shelf-pose-c arm))
      (:d (send self :avoid-shelf-pose-d arm))
      (:e (send self :avoid-shelf-pose-e arm))
      (:f (send self :avoid-shelf-pose-f arm))
      (:g (send self :avoid-shelf-pose-g arm))
      (:h (send self :avoid-shelf-pose-h arm))
      (:i (send self :avoid-shelf-pose-i arm))
      (:j (send self :avoid-shelf-pose-j arm))
      (:k (send self :avoid-shelf-pose-k arm))
      (:l (send self :avoid-shelf-pose-l arm)))
    )
  (:view-hand-pose
    (arm bin)
    (case bin
      (:a (send self :view-hand-pose-a arm))
      (:b (send self :view-hand-pose-b arm))
      (:c (send self :view-hand-pose-c arm))
      (:d (send self :view-hand-pose-d arm))
      (:e (send self :view-hand-pose-e arm))
      (:f (send self :view-hand-pose-f arm))
      (:g (send self :view-hand-pose-g arm))
      (:h (send self :view-hand-pose-h arm))
      (:i (send self :view-hand-pose-i arm))
      (:j (send self :view-hand-pose-j arm))
      (:k (send self :view-hand-pose-k arm))
      (:l (send self :view-hand-pose-l arm)))
    )
  (:visualize-path-planning
    (limb path &key (other-objects))
    (let ((robots (list (copy-object self))))
      (dolist (goal path)
        (let* ((robot (copy-object (car (reverse robots))))
               (av)
               )
          (setq av
                (send robot limb :inverse-kinematics goal
                      :rotation-axis t
                      :debug-view nil
                      :warnp nil
                      :dump-command nil))
          (when av
            (send-all (send robot :bodies) :set-color nil 0.2)
            (pushback robot robots)
            )
          )
        )
      (objects (append robots path other-objects)))
    )
  )
