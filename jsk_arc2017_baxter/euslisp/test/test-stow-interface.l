#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/stow-interface.l")

(init-unit-test)

(jsk_arc2017_baxter::stow-init)

(deftest test-stow-interface
  ;; :recognize-tote-boxes, :recognize-bin-boxes
  ;; (send *ti* :recognize-tote-boxes)
  ;; (send *ti* :recognize-bin-boxes)
  (send *ti* :recognize-bboxes :larm)
  (send *ti* :recognize-bboxes :rarm)

  ;; visualize-bins
  (let ((bins-vis (send *ti* :visualize-boxes :box-type :bin))
        (tote-vis (send *ti* :visualize-boxes :box-type :tote)))
    (assert (listp bins-vis))
    (assert (listp tote-vis))
    (objects (append (list *baxter*) bins-vis tote-vis))
    )

  ;; :move-arm-body->bin-overlook-pose
  (send *ti* :move-arm-body->tote-overlook-pose :larm)
  (send *ti* :move-arm-body->tote-overlook-pose :rarm)

  ;; :fold-pose-back
  (send *ti* :fold-pose-back)
  (send *ti* :fold-pose-back :rarm)
  (send *ti* :fold-pose-back :larm)
  )

(run-all-tests)
(exit)
