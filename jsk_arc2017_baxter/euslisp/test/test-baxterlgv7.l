#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/baxterlgv7.l")

(init-unit-test)

(setq *baxter* (instance jsk_arc2017_baxter::baxterlgv7-robot :init))

(deftest test-hand-methods
  (assert (eq (length (send *baxter* :hand :rarm :angle-vector #f(10 20))) 2)
          "returned angle-vector of right hand has invaild length")
  (assert (eq (length (send *baxter* :hand :larm :angle-vector #f(10 20 30))) 3)
          "returned angle-vector of left hand has invaild length")
  (dolist (style (list :opposed :spherical :cylindrical))
    (assert (eq (length (send *baxter* :hand-grasp-pre-pose :rarm style)) 2)
            ":hand-grasp-pre-pose of rarm returns angle-vector having invaild length")
    (assert (eq (length (send *baxter* :hand-grasp-pose :rarm style :angle 100)) 2)
            ":hand-grasp-pose of rarm returns angle-vector having invaild length")
    (assert (eq (length (send *baxter* :hand-grasp-pre-pose :larm style)) 3)
            ":hand-grasp-pre-pose of larm returns angle-vector having invaild length")
    (assert (eq (length (send *baxter* :hand-grasp-pose :larm style :angle 100)) 3)
            ":hand-grasp-pose of larm returns angle-vector having invaild length")))

(run-all-tests)
(exit)
