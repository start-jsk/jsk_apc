#!/usr/bin/env roseus
(require :unittest "lib/llib/unittest.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter.l")

(init-unit-test)

(setq *error-output* nil) ; to suppress to many debug info

(setq *tote* (make-cube 300 540 150 :pos #f(750 0 -330)))

(deftest test-ik-in-tote
  (let (robot)
    (setq robot (instance jsk_2016_01_baxter_apc::baxter-robot :init))
    (setq step-dist 50)
    (setq x (elt (send (send *tote* :worldcoords) :pos) 0))
    (setq y (elt (send (send *tote* :worldcoords) :pos) 1))
    (setq z (elt (send (send *tote* :worldcoords) :pos) 2))
    (setq dx (x-of-cube *tote*))
    (setq dy (y-of-cube *tote*))
    (setq dz (z-of-cube *tote*))
    (do ((tmp-x (+ (- x (/ dx 2)) (/ step-dist 2)) (+ tmp-x step-dist)))
      ((> tmp-x (+ x (/ dx 2))) t)
      (do ((tmp-y (+ (- y (/ dy 2)) (/ step-dist 2)) (+ tmp-y step-dist)))
        ((> tmp-y (+ y (/ dy 2))) t)
        (do ((tmp-z (+ (- z (/ dz 2)) (/ step-dist 2)) (+ tmp-z step-dist)))
          ((> tmp-z (+ z (/ dz 2))) t)
          (setq pos (float-vector tmp-x tmp-y tmp-z))
          (assert
            (and
              (send robot :rarm :inverse-kinematics
                    (make-coords :pos pos
                                 :rpy #f(0 0 0))
                    :use-gripper t
                    :rotation-axis :z)
              (send robot :larm :inverse-kinematics
                    (make-coords :pos pos
                                 :rpy #f(0 0 0))
                    :use-gripper t
                    :rotation-axis :z))
            "fail to solve ik")
          )
        )
      )
    )
  )

(run-all-tests)
(exit)
