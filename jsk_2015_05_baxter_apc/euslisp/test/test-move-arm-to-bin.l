;; vim: set ft=lisp:

(require :unittest "lib/llib/unittest.l")

(require :baxter "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter.l")
(require :util "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/util.l")

;; load *bin-cubes*
(require :bin-cubes "package://jsk_2015_05_baxter_apc/euslisp/test/data/bin-cubes.l")


(init-unit-test)


(setq *baxter* (instance jsk_2015_05_baxter_apc::baxter-robot :init))
(setq *robot* *baxter*)


(deftest
  test-move-arm-to-bin
  (dolist (bin (list :a :b :c :d :e :f :g :h :i :j :k :l))
    (case bin
      ((:a :b :d :e :g :h :j :k) (setq arm :larm))
      ((:c :f :i :l) (setq arm :rarm))
      )
    (let ((cube (gethash bin *bin-cubes*)))
      (dolist (offset (list #f(-150 0 0) #f(-50 0 0) #f(0 0 0) #f(0 0 -80)))
        (let ((goal (send cube :copy-worldcoords)))
          (send goal :translate offset :world)
          (assert (send *robot* arm :inverse-kinematics goal))
          )
        )
      )
    )
  )


(run-all-tests)
(exit)
