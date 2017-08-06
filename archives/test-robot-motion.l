#!/usr/bin/env roseus
(require :unittest "lib/llib/unittest.l")

(ros::roseus "test_robot_motion")

(require "package://jsk_2015_05_baxter_apc/euslisp/lib/baxter.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/main.l")

;; -------------------
;; Overwrite functions
;; -------------------
(defclass baxter-test-robot
  :super jsk_2015_05_baxter_apc::baxter-robot)
(defmethod baxter-test-robot
  (:inverse-kinematics
    (&rest args)
    (let (ret)
      (setq ret (send-super* :inverse-kinematics args))
      (unless ret (throw 'ik-error nil))
      ret)))
(defclass baxter-test-interface
  :super jsk_2015_05_baxter_apc::baxter-interface)
(defmethod baxter-test-interface
  (:wait-interpolation (&rest args) t)
  (:angle-vector-sequence (&rest args) t))
(defun unix::sleep (&rest args) t)

(apc-init)
(setq *baxter* (instance baxter-test-robot :init))
(setq *ri* (instance baxter-test-interface :init))

(init-unit-test)

;; left/right side is symmetric
(setq test-patterns (cons :larm '(:a :b :d :e :g :h :j :k)))
(setq test-names '("move-to-bin" "pick-object" "move-for-verification" "place-object"))

(deftest
  test-apc-inverse-kinematics
  (let ((arm (car test-patterns))
        (bins (cdr test-patterns)))
    (send *ri* :recognize-bin-boxes)
    (setq passed nil)
    (dolist (bin bins)
      (catch 'ik-error
             (progn (send *ri* :move-arm-body->bin arm bin)
                    (send *ri* :wait-interpolation))
             (push 0 passed))
      (catch 'ik-error
             (progn
               (send *ri* :pick-object arm bin)
               (send *baxter* :fold-to-keep-object arm)
               (send *ri* :send-av)
               (send *ri* :wait-interpolation))
             (push 1 passed))
      (catch 'ik-error
             (progn (send *ri* :move-arm-body->head-view-point arm)
                    (send *ri* :wait-interpolation))
             (push 2 passed))
      (catch 'ik-error
             (progn (send *ri* :move-arm-body->order-bin arm)
                    (send *ri* :wait-interpolation))
             (push 3 passed))
      (dotimes (i (length test-names))
        (unless (member i passed)
          (assert nil (format nil "ik failed in ~A ~A ~A" (elt test-names i) arm bin))))
    )))

(run-all-tests)
(exit)
