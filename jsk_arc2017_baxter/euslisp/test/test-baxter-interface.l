#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/baxter-interface.l")

(init-unit-test)

(setq *ri* (instance jsk_arc2017_baxter::baxter-interface :init))

(deftest test-move-hand
  (send *ri* :move-hand :larm #f(10 20) 1000 :wait nil)
  (send *ri* :move-hand :rarm #f(10 20) 1000 :wait nil))

(run-all-tests)
(exit)
