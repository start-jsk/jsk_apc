(load "package://baxtereus/baxter-interface.l")


(defun init ()
  (baxter-init)
  (send *baxter* :locate #f(0 0 950) :world)
  (objects (list *baxter*)))


(defun test-1 ()
  (send *baxter* :larm :shoulder-y :joint-angle -30)
  (send *baxter* :larm :shoulder-p :joint-angle -30)
  (send *baxter* :larm :elbow-r  :joint-angle 30)
  (send *baxter* :larm :elbow-p  :joint-angle 90)
  (send *baxter* :larm :wrist-r  :joint-angle 30)
  (send *baxter* :larm :wrist-p  :joint-angle -30)
  (send *baxter* :head :neck-y   :joint-angle -30)
  (objects (list *baxter*))
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  )


(defun test-2 ()
  (let ()
    (setq *box* (make-cube 100 100 200))
    (send *box* :set-color :blue)
    (do ((z 1100 (+ z 100)))
        ((>= z 1500))
        (do ((y 0 (+ y 100)))
            ((>= y 500))
            (print (list 900 y z))
            (send *baxter* :reset-pose)
            (send *box* :locate (float-vector 900 y z) :world)
            (objects (list *baxter* *box*))
            (send *baxter* :larm :inverse-kinematics *box* :rotation-axis :z :debug-view :no-message)
            (send *baxter* :look-at-hand :larm)
            (send *irtviewer* :draw-objects)
            (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
            (send *ri* :wait-interpolation)
            ))
    ))


(defclass target-pick-region
  :super propertied-object
  :slots (a b c d e f g h i j k l))
(defmethod target-pick-region
  (:init ()
    """Local coordinates of each region
    in the podlowres objects"""
    (setq a #f(-270 1660 500))
    (setq b #f(   0 1660 500))
    (setq c #f( 270 1660 500))
    (setq d #f(-270 1430 500))
    (setq e #f(   0 1430 500))
    (setq f #f( 270 1430 500))
    (setq g #f(-270 1200 500))
    (setq h #f(   0 1200 500))
    (setq i #f( 270 1200 500))
    (setq j #f(-270  950 500))
    (setq k #f(   0  950 500))
    (setq l #f( 270  950 500)))
  (:target-list ()
    (list a b c d e f g h i j k l)))


(defun test-target-ik ()
  """Target pick region test
  to confirm the endpoint can reached
  """
  (let (i region c b r)
    ;; prepare pod-lowres
    (when (not (boundp '*pod*))
      (load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l"))
    (send *pod* :locate #f(1300 0 0) :world)
    (send *pod* :rotate pi/2 :x :world)
    (send *pod* :rotate (- pi/2) :z :world)
    (objects (list *baxter* *pod*))
    ;; make target cube
    (setq *targets* nil i 0)
    (setq region (instance target-pick-region :init))
    (dolist (c (send region :target-list))
      ;; transform target coordinates local->world
      (setq c (send *pod* :transform-vector c))
      (setq b (make-cube 25 25 100 :pos c))
      (send b :set-color (elt gl::*face-colors* (* i 10)))
      (objects b)
      ;; set cube as target
      (set (intern (format nil "TARGET_~A" i)) b)
      (push (eval (intern (format nil "TARGET_~A" i))) *targets*)
      (incf i)
      )
    ;; calc inverse kinematics on each target
    ;; if success with z constraction -> cube is red
    ;; elif success with no constraction -> magenta
    (dolist (target *targets*)
      ;; ik with z axis constraction
      (setq r (send *baxter* :larm :inverse-kinematics target
                    :rotation-axis :z
                    :debug-view :no-message))
      (if r (send target :set-color :red)
        (progn
          ;; ik without axis constraction
          (setq r (send *baxter* :larm :inverse-kinematics target
                        :rotation-axis nil
                        :debug-view :no-message))
          (if r (send target :set-color :magenta))))
        (send *irtviewer* :draw-objects)
        (x::window-main-one)
        )
    ))

(warn "~%")
(warn "(init) to initialize baxter model.~%")
(warn "(test-1) to move baxter arm a little.~%")
(warn "(test-2) to test simple IK.~%")
(warn "(test-target-ik) to test target regeion IK.~%")

