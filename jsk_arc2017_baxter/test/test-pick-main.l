#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/pick-main.l")

(init-unit-test)

(deftest test-pick-main
   (make-picking-state-machine)
   (assert (boundp '*sm*)
           "failed to set picking-state-machine *sm*"))

(run-all-tests)
(exit)
