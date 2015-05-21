#!/usr/bin/env roseus
(require :unittest "lib/llib/unittest.l")

(ros::roseus "test_robot_motion")

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")

;; -------------------
;; Overwrite functions
;; -------------------
(defclass baxter-test-robot
  :super baxter-robot)
(defmethod baxter-test-robot
  (:inverse-kinematics
    (&rest args)
    (let (ret)
      (setq ret (send-super* :inverse-kinematics args))
      (unless ret (throw 'ik-error nil))
      ret)))
(defclass baxter-test-interface
  :super baxter-interface)
(defmethod baxter-test-interface
  (:wait-interpolation (&rest args) t)
  (:angle-vector-sequence (&rest args) t))
(defun unix::sleep (&rest args) t)

(apc-init)
(setq *baxter* (instance baxter-test-robot :init))
(send *baxter* :locate #f(0 0 950) :world)
(send (send *baxter* :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
(send (send *baxter* :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
(setq *ri* (instance baxter-test-interface :init))

(defclass baxter-test-robot
  :super baxter-robot)
(defmethod baxter-test-robot
  (:inverse-kinematics (&rest args)
   (let (ret)
    (setq ret (send-super* :inverse-kinematics args))
    (assert (null ret) "ik failed")
    ret)))

(init-unit-test)

(deftest
  test-move-to-target-bin
  (dolist (bin '(:a :b :c :d :e :f :g :h :i :j :k :l))
    (case (which-bin-region bin)
      (:left :center (setq arm :larm))
      (:right (setq arm :rarm)))
    (setq test-passed nil)
    (catch 'ik-error (move-to-target-bin arm bin) (setq test-passed t))
    (assert test-passed (format nil "ik failed in move-to-target-bin: ~A ~A" arm bin))
    ))

(deftest
  test-pick-object
  (dolist (bin '(:a :b :c :d :e :f :g :h :i :j :k :l))
    (case (which-bin-region bin)
      (:left :center (setq arm :larm))
      (:right (setq arm :rarm)))
    (setq test-passed nil)
    (move-to-target-bin arm bin)
    (catch 'ik-error (pick-object arm bin) (setq test-passed t))
    (assert test-passed (format nil "ik failed in pick-object ~A ~A" arm bin))
    ))

(deftest
  test-move-for-verification
  (dolist (bin '(:a :b :c :d :e :f :g :h :i :j :k :l))
    (case (which-bin-region bin)
      (:left :center (setq arm :larm))
      (:right (setq arm :rarm)))
    (setq test-passed nil)
    (move-to-target-bin arm bin)
    (pick-object arm bin)
    (catch 'ik-error (move-for-verification arm bin) (setq test-passed t))
    (assert test-passed (format nil "ik failed in move-for-verification ~A ~A" arm bin))
    ))

(deftest
  test-place-object
  (dolist (bin '(:a :b :c :d :e :f :g :h :i :j :k :l))
    (case (which-bin-region bin)
      (:left :center (setq arm :larm))
      (:right (setq arm :rarm)))
    (setq test-passed nil)
    (move-to-target-bin arm bin)
    (pick-object arm bin)
    (move-for-verification arm bin)
    (catch 'ik-error (place-object arm) (setq test-passed t))
    (assert test-passed (format nil "ik failed in place-object ~A ~A" arm bin))
    ))

(run-all-tests)
(exit)
