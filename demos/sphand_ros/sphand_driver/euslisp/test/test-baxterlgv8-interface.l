#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://sphand_driver/euslisp/lib/baxterlgv8-interface.l")

(init-unit-test)

(setq *ri* (instance sphand_driver::baxterlgv8-interface :init))

(deftest test-move-hand
  (send *ri* :move-hand :larm #f(10 20 30) 1000 :wait nil)
  (send *ri* :move-hand :rarm #f(10 20) 1000 :wait nil))

(run-all-tests)
(exit)
