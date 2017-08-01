#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/stow-main.l")

(init-unit-test)

(deftest test-stow-main
   (make-stowing-state-machine)
   (assert (boundp '*sm*)
           "failed to set stowing-state-machine *sm*"))

(run-all-tests)
(exit)
