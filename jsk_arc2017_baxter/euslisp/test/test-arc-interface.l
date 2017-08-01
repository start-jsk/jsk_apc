#!/usr/bin/env roseus

(require :unittest "lib/llib/unittest.l")
(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(init-unit-test)

(jsk_arc2017_baxter::arc-init)

(deftest test-get-next-target-bin
  (setq orders nil)
  (dolist (item (list "expo_eraser" "avery_binder"))
    (setq msg (instance jsk_arc2017_common::WorkOrder :init))
    (send msg :bin (if (string= item "expo_eraser") "A" "C"))
    (send msg :item item)
    (setq orders (append orders (list msg))))
  (setq (*ti* . orders-) orders)
  (assert (find (send *ti* :get-next-target-bin :rarm) (list :a :c))
          (format nil "failed to get-next-target-bin: ~a, expecting (list :a :c)"
                  (symbol2str (send *ti* :get-next-target-bin :rarm))))
  (send *ti* :add-finished-object :rarm "expo_eraser" :a)
  (send *ti* :add-postponed-object :rarm "avery_binder" :c)
  (assert (eq (send *ti* :get-next-target-bin :rarm) :c)
          (format nil "failed to get-next-target-bin: ~a,  expecting :c"
                  (symbol2str (send *ti* :get-next-target-bin :rarm))))
  (send *ti* :add-finished-object :rarm "avery_binder" :c)
  (assert (null (send *ti* :get-next-target-bin :rarm))
          (format nil "failed to get-next-target-bin: ~a,  expecting null"
                  (symbol2str (send *ti* :get-next-target-bin :rarm)))))

(ros::service-call "~set_logger_level"
                   (instance roscpp::SetLoggerLevelRequest
                             :init :logger "ros.roseus" :level "DEBUG"))
(run-all-tests)
(exit)
