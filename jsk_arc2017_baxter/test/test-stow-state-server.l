#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/stow-interface.l")

(init-unit-test)

(deftest test-stow-state-server
  (jsk_arc2017_baxter::stow-init :moveit nil)
  (assert
    (send *ti* :update-state :rarm :wait-for-opposite-arm)
    "failed to update-state :rarm")
  (assert
    (send *ti* :update-state :larm :wait-for-opposite-arm)
    "failed to update-state :larm")
  (assert
    (send *ti* :check-can-start :rarm :recognize-object :wait-for-opposite-arm)
    "failed to :check-can-start :rarm")
  (assert
    (null (send *ti* :check-can-start :larm :recognize-object :wait-for-opposite-arm))
    "failed to :check-can-start :larm")
  (assert
    (string= (send *ti* :get-state :rarm) "recognize-object")
    "failed to :get-state :rarm")
  (assert
    (string= (send *ti* :get-state :larm) "wait-for-opposite-arm")
    "failed to :get-state :larm"))

(run-all-tests)
(exit)
