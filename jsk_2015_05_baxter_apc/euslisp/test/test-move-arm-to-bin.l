;; vim: set ft=lisp:

(require :unittest "lib/llib/unittest.l")

(require :baxter "package://jsk_2015_05_baxter_apc/euslisp/lib/baxter.l")
(require :util "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")

;; load *bin-cubes-alist*
(require :bin-cubes-alist "package://jsk_2015_05_baxter_apc/euslisp/test/data/bin-cubes-alist.l")


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
    (let ((cube (cdr (assoc bin *bin-cubes-alist*))))
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
