#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(init-unit-test)

(jsk_arc2017_baxter::arc-init)

(deftest test-get-next-target-bin
  (setq orders nil)
  (dolist (item (list "a" "b"))
    (setq msg (instance jsk_arc2017_common::WorkOrder :init))
    (send msg :bin (if (string= item "a") "A" "C"))
    (send msg :item item)
    (setq orders (append orders (list msg))))
  (setq (*ti* . orders-) orders)
  (assert (find (send *ti* :get-next-target-bin :rarm) (list :a :c))
          (format nil "failed to get-next-target-bin: ~a"
                  (symbol2str (send *ti* :get-next-target-bin :rarm))))
  (send *ti* :add-finished-object :rarm "a")
  (send *ti* :add-postponed-object :rarm "b")
  (assert (eq (send *ti* :get-next-target-bin :rarm) :c)
          (format nil "failed to get-next-target-bin: ~a"
                  (symbol2str (send *ti* :get-next-target-bin :rarm))))
  (send *ti* :add-finished-object :rarm "b")
  (assert (null (send *ti* :get-next-target-bin :rarm))
          (format nil "failed to get-next-target-bin: ~a"
                  (symbol2str (send *ti* :get-next-target-bin :rarm)))))

(run-all-tests)
(exit)
