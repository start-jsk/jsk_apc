#!/usr/bin/env roseus
(require :unittest "lib/llib/unittest.l")

(load "package://jsk_2014_picking_challenge/euslisp/robot-test.l")

(init-unit-test)

(deftest test-robot-motion
         (dolist (bin (list :a :b :c :d :e :f :g :h :i :j :k :l))
           (dolist (arm (list :larm :rarm))
             (move-to-target-bin arm bin)
             (pick-object arm)
             (move-for-verification arm)
             (place-object arm))))
(run-all-tests)
(exit)
