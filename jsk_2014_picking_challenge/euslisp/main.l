#!/usr/bin/env roseus
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/arm-motion.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")
(load "package://jsk_2014_picking_challenge/euslisp/gripper-control.l")


(defun init ()
  (baxter-init)
  (send *baxter* :locate #f(0 0 950) :world)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (pod-init)
  (orderbin-init)
  (objects (list *baxter* *pod* *orderbin*)))

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
  (send *irtviewer* :draw-objects))

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

(defun test-target-ik (&optional (insert 0))
  "Target pick region test
  to confirm the endpoint can reached"
  (let (i region c b r)
    ;; prepare pod-lowres
    (pod-init)
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
    ;; if success without axis free -> red
    ;; elif success with z axis free -> magenta
    (dolist (target *targets*)
      ;; ik without axis free
      (setq r (send *baxter* :larm :inverse-kinematics target
                    :rotation-axis t
                    :debug-view :no-message))
      (setq r (and r (send *baxter* :larm :move-end-pos (float-vector insert 0 0) :world)))
      (if r (send target :set-color :red)
        (progn
          ;; ik with z axis free
          (setq r (send *baxter* :larm :inverse-kinematics target
                        :rotation-axis :z
                        :debug-view :no-message))
          (setq r (and r (send *baxter* :larm :move-end-pos (float-vector insert 0 0) :world)))
          (if r (send target :set-color :magenta))))
        (send *irtviewer* :draw-objects)
        (x::window-main-one)
        )
    ))

(defun ir2ri (&optional (tm nil))
  (send *ri* :angle-vector (send *baxter* :angle-vector) tm)
  (send *irtviewer* :draw-objects))

(defun ri2ir ()
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *irtviewer* :draw-objects))

(defun mv2target (arm bin_nm)
  "Move arm to bin"
  (let (region c target)
    ;; validation
    (setq region (instance target-pick-region :init))
    (unless (find bin_nm (send region :target-name-list))
      (warn "unknown bin name")
      (return-from mv2target))
    (setq c (send *pod* :transform-vector (send region :get-val bin_nm)))
    (setq target (instance cascaded-coords :init :pos c))
    (unless (find arm (list :larm :rarm))
      (warn "specify :larm or :rarm")
      (return-from mv2target))
    ;; inverse kinematics
    (setq r (send *baxter* arm :inverse-kinematics target :rotation-axis t))
    (when (not r)
      (warn "ik with z axis free~%")
      (setq r (send *baxter* arm :inverse-kinematics target :rotation-axis :z))
      (when (not r)
        (warn "failed to compute IK~%")
        (return-from mv2target)))
    ;; reflect change to ri
    (send *irtviewer* :draw-objects)
    (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    ))

(warn "~%")
(warn "(test-1) to move baxter arm a little.~%")
(warn "(test-2) to test simple IK.~%")
(warn "(test-target-ik) to test target regeion IK.~%")
(warn "(ir2ri) to reflect ir angle vector to ri.~%")
(warn "(ri2ir) to reflect potentio vector to ir viewer.~%")
(warn "(mv2target arm bin_nm) to move arm (ex. :larm) to target bin (ex. 'a)~%")
(init)
