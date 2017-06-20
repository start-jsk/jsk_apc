#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(init-unit-test)

(jsk_arc2017_baxter::arc-init)

(deftest test-pick-arc-interface
  ;; :recognize-bin-boxes
  (send *ti* :recognize-bin-boxes)

  ;; visualize-bins
  (let ((bins-vis (send *ti* :visualize-boxes :box-type :bin)))
    (assert (listp bins-vis))
    (objects (append (list *baxter*) bins-vis))
    )

  ;; :move-arm-body->bin-overlook-pose
  (send *ti* :move-arm-body->bin-overlook-pose :larm :a)
  (send *ti* :move-arm-body->bin-overlook-pose :rarm :b)
  (send *ti* :move-arm-body->bin-overlook-pose :rarm :c)

  ;; :fold-pose-back
  (send *ti* :fold-pose-back)
  (send *ti* :fold-pose-back :rarm)
  (send *ti* :fold-pose-back :larm)
  )

(run-all-tests)
(exit)
