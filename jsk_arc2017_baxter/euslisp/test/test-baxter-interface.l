#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/baxter-interface.l")

(init-unit-test)

(setq *ri* (instance jsk_arc2017_baxter::baxter-interface :init))

(run-all-tests)
(exit)
