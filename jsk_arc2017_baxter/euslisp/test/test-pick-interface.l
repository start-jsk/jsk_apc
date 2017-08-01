#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/pick-interface.l")

(init-unit-test)

(jsk_arc2017_baxter::pick-init)

(deftest test-pick-interface
  ;; :recognize-bin-boxes, :recognize-cardboard-boxes
  ;; (send *ti* :recognize-bin-boxes)
  ;; (send *ti* :recognize-cardboard-boxes)
  (send *ti* :recognize-bboxes)

  ;; visualize-bins
  (let ((bins-vis (send *ti* :visualize-boxes :box-type :bin))
        (cardboard-boxes-vis (send *ti* :visualize-boxes :box-type :cardboard)))
    (assert (listp bins-vis))
    (assert (listp cardboard-boxes-vis))
    (objects (append (list *baxter*) bins-vis cardboard-boxes-vis))
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
