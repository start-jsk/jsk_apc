(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxterrgv5-interface.l")

(defun demo-init ()
  (jsk_2016_01_baxter_apc::baxterrgv5-init :ctype :default-controller)
  (send *baxter* :fold-pose-back)
  (objects (list *baxter*))
  (setq now (ros::time-now))
  (send *ri* :recognize-bin-boxes :stamp now)
  (send *irtviewer* :change-background #f(1 1 1))
  t)

(defun make-bin-and-object
  (&key (bin :c) (object-size (list 50 50 50)))
  (setq bin-boxes (send *ri* :get-val '_bin-boxes))
  (setq bin-box (gethash bin bin-boxes))
  (setq bin-cube (send *ri* :bbox->cube bin-box))
  (setq bin-cube-base (send *ri* :cube->cube-parallel-to-coords
                            bin-cube
                            :included-in-original-cube))
  (setq wall-thickness 30)
  (setq bin-wall (make-cube
                   (+ (x-of-cube bin-cube-base) wall-thickness)
                   (+ (y-of-cube bin-cube-base) (* wall-thickness 2))
                   (+ (z-of-cube bin-cube-base) wall-thickness)))
  (setq bin-floor (make-cube
                    (+ (x-of-cube bin-cube-base) wall-thickness)
                    (+ (y-of-cube bin-cube-base) (* wall-thickness 2))
                    wall-thickness))
  (send bin-wall :move-to bin-cube-base :world)
  (send bin-floor :move-to bin-cube-base :world)
  (send bin-wall :locate (float-vector (/ wall-thickness 2) 0 (/ wall-thickness 2)) :local)
  (send bin-floor :locate
        (float-vector
          (/ wall-thickness 2) 0 (/ (+ (z-of-cube bin-cube-base) wall-thickness) -2))
        :local)
  (setq bin-wall (body- bin-wall bin-cube-base))
  (send bin-wall :set-color #f(0 0.3 1) 0.3)
  (send bin-floor :set-color #f(0 0.3 1) 1)

  (setq object-cube (apply #'make-cube object-size))
  (send object-cube :set-color #f(1 0 0))
  (send object-cube :move-to bin-cube-base :world)
  (objects (list object-cube *baxter* bin-floor bin-wall))
  )

(defun locate-object
  (&key (object-offset #f(0 0 0)) (on-floor nil))
  (send object-cube :move-to bin-cube-base :world)
  (send object-cube :locate
        (v+ (float-vector (/ (x-of-cube bin-cube-base) -2) 0 0) object-offset)
        :local)
  (if on-floor
    (send object-cube :locate
          (float-vector
            0 0
            (- (/ (z-of-cube object-cube) 2)
               (/ (z-of-cube bin-cube-base) 2)
               (elt object-offset 2)))
          :local)
    )
  (send *irtviewer* :draw-objects)
  )

(defun display-hand-poses
  (&key (sign_x 0) (sign_y 0) (style :cylindrical) (closep nil) object-size object-offset
        (robot *baxter*))
  (if style
    (send robot :hand-grasp-pre-pose :rarm style)
    (progn
      (send robot :hand-grasp-pre-pose :rarm :opposed)
      (setq style :opposed)
      (setq closep t)
      )
    )
  (if closep
    (progn
      (if (null object-size) (setq object-size (list 60 60 60)))
      ;; (list 60 60 80) is recommended when sign_y = -1
      (if (null object-offset) (setq object-offset #f(200 0 0)))
      (unless (and (= (x-of-cube object-cube) (elt object-size 0))
                   (= (y-of-cube object-cube) (elt object-size 1))
                   (= (z-of-cube object-cube) (elt object-size 2)))
        (make-bin-and-object :object-size object-size)
        )
      (locate-object :object-offset object-offset :on-floor t)
      (setq target-coords (send object-cube :copy-worldcoords))
      (send target-coords :rotate (* (deg2rad 90) sign_x) :x :world)
      (send target-coords :rotate (* (deg2rad 90) sign_y) :y :world)
      (send robot :slide-gripper :rarm 124 :relative nil)
      (send robot :hand-grasp-pose :rarm style :angle 90)
      (if (< sign_y 0)
        (progn
          (send target-coords :translate #f(0 0 -30) :world)
          (send robot :rotate-gripper :rarm 0 :relative nil)
          )
        (send robot :rotate-gripper :rarm 90 :relative nil)
        )
      (send robot :rarm :inverse-kinematics target-coords)
      (send robot :rarm :move-end-pos #f(0 0 35) :local)
      (send *irtviewer* :draw-objects)
      (unless (y-or-n-p "continue?") (return-from display-hand-poses))
      (if (< sign_y 0)
        (send robot :hand-grasp-pose :rarm style :angle 25)
        (progn
          (send (send robot :rarm :end-coords) :assoc object-cube)
          (send robot :rarm :move-end-pos #f(-160 0 0) :world)
          (send (send robot :rarm :end-coords) :dissoc object-cube)
          )
        )
      (send *irtviewer* :draw-objects)
      )
    (progn
      (if (null object-size) (setq object-size (list 80 80 80)))
      (if (null object-offset) (setq object-offset #f(90 0 0)))
      (unless (and (= (x-of-cube object-cube) (elt object-size 0))
                   (= (y-of-cube object-cube) (elt object-size 1))
                   (= (z-of-cube object-cube) (elt object-size 2)))
        (make-bin-and-object :object-size object-size)
        )
      (locate-object :object-offset object-offset :on-floor t)
      (setq target-coords (send object-cube :copy-worldcoords))
      (send target-coords :rotate (* (deg2rad 90) sign_x) :x :world)
      (send target-coords :rotate (* (deg2rad 90) sign_y) :y :world)
      (if (< sign_y 0)
        (send target-coords :translate #f(0 0 -30) :world)
        (send target-coords :rotate (deg2rad 180) :z :world))
      (if (< sign_y 0)
        (progn
          (send robot :slide-gripper :rarm 50 :relative nil)
          (send robot :rotate-gripper :rarm 0 :relative nil)
          )
        (progn
          (send robot :slide-gripper :rarm 100 :relative nil)
          (send robot :rotate-gripper :rarm -90 :relative nil)
          )
        )
      (send robot :rarm :inverse-kinematics target-coords)
      (send robot :rarm :move-end-pos #f(0 0 45) :local)
      (send *irtviewer* :draw-objects)
      (unless (y-or-n-p "continue?") (return-from display-hand-poses))
      (send (send robot :rarm :end-coords) :assoc object-cube)
      (send robot :rarm :move-end-pos #f(-200 0 0) :world)
      (if (eq style :cylindrical)
        (send robot :hand-grasp-pose :rarm style :angle 37)
        (progn
          (send robot :hand-grasp-pose :rarm style :angle 70)
          (send robot :slide-gripper :rarm 0 :relative nil)
          )
        )
      (send (send robot :rarm :end-coords) :dissoc object-cube)
      (send *irtviewer* :draw-objects)
      )
    )
  )
