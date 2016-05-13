#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require :baxter-interface "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")

(init-unit-test)

(setq *error-output* nil) ; to suppress to many debug info

(deftest
  test-move-arm-to-bin
  (jsk_2016_01_baxter_apc::baxter-init)
  (send *ri* :recognize-bin-boxes :stamp (ros::time 0))
  (dolist (bin (list :a :b :c :d :e :f :g :h :i :j :k :l))
    (case bin
      ((:a :d :g :j) (setq arm-list (list :larm)))
      ((:b :e :h :k) (setq arm-list (list :larm :rarm)))
      ((:c :f :i :l) (setq arm-list (list :rarm)))
      )
    (dolist (arm arm-list)
      (dolist (offset (list #f(-150 0 0) #f(-50 0 0) #f(0 0 0) #f(0 0 -80)))
        (assert (send *ri* :ik->bin-entrance arm bin :offset offset :revert-if-fail t))
        )
      )
    )
  )

(run-all-tests)
(exit)
