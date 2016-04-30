#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require :baxter-interface "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")

(init-unit-test)

(deftest
  test-move-arm-to-bin
  (jsk_2016_01_baxter_apc::baxter-init)
  (send *ri* :recognize-bin-boxes :stamp (ros::time 0))
  (assert (send *ri* :ik->bin-entrance :rarm :e))
  )

(run-all-tests)
(exit)
