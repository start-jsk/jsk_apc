;; vim: set ft=lisp:
(require "package://synthetic2d/euslisp/lib/stow-interface.l")
(ros::load-ros-manifest "synthetic2d")

(defun use-arm (arm)
  (when arm
    (setq *arm* arm)
    )
  )

(unless (boundp '*arm*)
  (use-arm :larm)
  )

(defun init (&optional (arm nil))
  (use-arm arm)
  (synthetic2d::stow-init)
  (setq *robot* *baxter*)
  (send *ri* :calib-pressure-threshold *arm*)
  )

;; TODO(wkentaro): Support right arm.

(defun setup()
  (send *ti* :recognize-tote-boxes)

  ;; reset the pose
  (send *baxter* :untuck-pose)

  ;; rarm no collision pose
  (setq rav #f(-40.9131 5.3833 6.43799 83.4741 3.2959 -1.99951 -12.1729 0.699495 1.43141))
  (if (eq *arm* :larm)
    (send *baxter* :rarm :angle-vector rav)
    (send *baxter* :larm :angle-vector (send *baxter* :l/r-reverse rav))
    )

  ;; tote-overlook-pose-refined
  ; (send *baxter* :larm :angle-vector #f(-16.5234 -26.6748 -90.4175 105.842 66.8848 86.9897 -76.4648 0.809942 0.0))
  (send *ri* :move-hand *arm* (send *baxter* :hand-grasp-pose *arm* :cylindrical :angle 90) 1000 :wait nil)
  (let (c)
    (setq c (send (gethash *arm* (*ti* . tote-cubes-)) :copy-worldcoords))
    (send c :translate (float-vector -150 0 200) :world)
    (send *baxter* *arm* :inverse-kinematics c :rotation-axis t)
    )

  ; (objects (list *baxter*))

  (send *ti* :send-av)
  (send *ri* :wait-interpolation)
  )

(defun pick (&key (discriminate t))
  (let (done)
    (setup)

    (setq avs nil)

    (pushback (send *baxter* :angle-vector) avs)

    (setq stamp (send *ti* :recognize-target-object *arm* :timeout 20))
    (when (null stamp) (return-from pick t)) ;; done
    (gethash *arm* (*ti* . object-boxes-))
    (print stamp)
    (if discriminate
      (progn
        (setq req (instance synthetic2d::IsTargetRequest :init))
        (send req :stamp stamp)
        (if (eq *arm* :larm)
          (setq res (ros::service-call "/left_hand_camera/mask_rcnn_relook/is_target" req))
          (setq res (ros::service-call "/right_hand_camera/mask_rcnn_relook/is_target" req))
          )
        (setq is_target (send res :is_target))
        )
      (progn
        (setq is_target t)
        )
      )

    (setq avoid-tote t)
    (unless is_target
      (setq obj-pbox (first (gethash *arm* (*ti* . object-pboxes-))))
      (if obj-pbox
        (setq avoid-tote nil)
        )
      )

    (setq (*ti* . target-obj-) "an_object")
    (setq (*ti* . grasp-style-) :suction)
    (setq graspingp
          (send *ti* :pick-object
                *arm*
                :use-scale nil
                :move-head nil
                :avoid-tote avoid-tote
              )
        )
    (send *ri* :wait-interpolation)

    (unless graspingp
      (send *ri* :stop-grasp *arm*)
      (send *ri* :angle-vector-sequence (reverse avs) :fast nil 0 :scale 5.0)
      (send *ri* :wait-interpolation)
      (return-from pick nil)
      )

    (pushback (send *baxter* :angle-vector) avs)

    (setq c (send (gethash *arm* (*ti* . tote-cubes-)) :copy-worldcoords))
    (if is_target
      (progn
        (ros::ros-info-green "It is the target!")
        (send *ri* :speak "It is the target!")
        (setq avs_place nil)
        (send c :translate (float-vector 400 (+ (* (random 3) -100) 100) 300) :world)
        (pushback (send *baxter* *arm* :inverse-kinematics c :rotation-axis nil) avs_place)
        (send c :translate (float-vector 0 0 -200) :world)
        (pushback (send *baxter* *arm* :inverse-kinematics c :rotation-axis nil) avs_place)
        (send *ri* :angle-vector-sequence avs_place :fast nil 0 :scale 5.0)
        (setq avs (append avs avs_place))
        (if discriminate
          (progn (setq done t))
          (progn (setq done nil))
          )
        )
      (progn
        (ros::ros-info-red "It is not the target!")
        (send *ri* :speak "It is not the target!")
          (if obj-pbox
            (progn
              (setq obj-pcube (send *ti* :bbox->cube obj-pbox))
              (setq obj-box (first (gethash *arm* (*ti* . object-boxes-))))
              (setq obj-cube (send *ti* :bbox->cube obj-box))
              (setq c (send obj-pcube :copy-worldcoords))
              (send c :translate (float-vector 0 0 100) :world)
              (send c :transform (send (make-coords :rot (send obj-cube :rot)) :inverse-transformation))
              (send *baxter* *arm* :inverse-kinematics c :rotation-axis t)
              (send *ri* :angle-vector (send *baxter* :angle-vector) 5000)
              )
            (progn
              (send c :translate (float-vector 0 -500 100) :world)
              (send *baxter* *arm* :inverse-kinematics c :rotation-axis nil)
              (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
              )
            )
          (setq done nil)
          )
        )

    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp *arm*)
    (unix::sleep 4)

    (send *ri* :angle-vector-sequence (reverse avs) :fast nil 0 :scale 5.0)
    (send *ri* :wait-interpolation)
  done))

(defun upick()
  (pick :discriminate nil)  ;; undiscriminate
  )

(defun pick-pick()
  (send *ri* :speak "I begin picking a target from a pile of objects!")
  (while (not (pick)))
  (send *ri* :speak "I have done the picking task!")
  )

(defun upick-upick()
  (send *ri* :speak "I begin undiscrimiate picking from a pile of objects!")
  (while (not (upick)))
  (send *ri* :speak "I have done the picking task!")
  )
