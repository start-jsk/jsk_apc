#!/usr/bin/env roseus
(require :unittest "lib/llib/unittest.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/test/data/bin-cubes.l")

(init-unit-test)

(setq *bin-keys* (list :a :b :c :d :e :f :g :h :i :j :k :l))
(setq *pod-bin-ik-limit*
      (list 1200 1000 1100 1300 1200 1300 0 1300 1300 1300 1300 1300))

(setq *error-output* nil) ; to suppress to many debug info

(deftest test-ik-in-bin
  (let (robot)
    (setq robot (instance jsk_2016_01_baxter_apc::baxter-robot :init))
    (setq step-dist 100)
    (dolist (cube-data (mapcar #'list *pod-bin-ik-limit* *bin-keys*))
      (setq limit (car cube-data))
      (setq key (cadr cube-data))
      (setq cube (gethash key *bin-cubes*))
      (setq x (elt (send (send cube :worldcoords) :pos) 0))
      (setq y (elt (send (send cube :worldcoords) :pos) 1))
      (setq z (elt (send (send cube :worldcoords) :pos) 2))
      (setq dx (x-of-cube cube))
      (setq dy (y-of-cube cube))
      (setq dz (z-of-cube cube))
      (send robot :reset-pose)
      (do ((tmp-x (+ (- x (/ dx 2)) (/ step-dist 2)) (+ tmp-x step-dist)))
        ((> tmp-x (+ x (/ dx 2))) t)
        (do ((tmp-y (+ (- y (/ dy 2)) (/ step-dist 2)) (+ tmp-y step-dist)))
          ((> tmp-y (+ y (/ dy 2))) t)
          (do ((tmp-z (+ (- z (/ dz 2)) (/ step-dist 2)) (+ tmp-z step-dist)))
            ((> tmp-z (+ z (/ dz 2))) t)
            (setq pos (float-vector tmp-x tmp-y tmp-z))
            (assert
              (or
                (send robot :rarm :inverse-kinematics
                  (make-cascoords :pos pos)
                  :rotation-axis t
                  :debug-view nil
                  :warnp nil
                  :dump-command nil)
                (send robot :larm :inverse-kinematics
                  (make-cascoords :pos pos)
                  :rotation-axis t
                  :debug-view nil
                  :warnp nil
                  :dump-command nil)
                (> tmp-x limit))
              "fail to solve ik")
            )
          )
        )
      )
    )
  )

(run-all-tests)
(exit)
