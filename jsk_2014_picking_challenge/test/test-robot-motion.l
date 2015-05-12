#!/usr/bin/env roseus
(require :unittest "lib/llib/unittest.l")

(load "package://jsk_2014_picking_challenge/euslisp/robot-test.l")

(init-unit-test)

(setq target-bins '(:a :b :c :d :e :f :g :h :i :j :k :l))

(deftest
  test-move-to-target-bin
  (dolist (bin target-bins)
    (cond ((find bin '(:a :d :g :j)) (setq arm :larm))
          ((find bin '(:c :f :i :l)) (setq arm :rarm))
          (t (setq arm :larm)))
    (assert (move-to-target-bin-av arm bin) "Failed move_to_target_bin")
    (send *irtviewer* :draw-objects)
    (unix::sleep 1)
    )
  )

; (deftest test-robot-motion
;          (dolist (bin (list :a :b :c :d :e :f :g :h :i :j :k :l))
;            (dolist (arm (list :larm :rarm))
;              (move-to-target-bin arm bin)
;              (pick-object arm)
;              (move-for-verification arm)
;              (place-object arm))))

(run-all-tests)
(exit)
