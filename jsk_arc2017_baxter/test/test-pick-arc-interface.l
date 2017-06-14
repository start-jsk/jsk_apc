#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(init-unit-test)

(jsk_arc2017_baxter::arc-init)

(deftest test-arc-interface-pick
  (send *ti* :recognize-bin-boxes)
  (objects (append (list *baxter*) (send *ti* :visualize-bins)))
  )

(run-all-tests)
(exit)
