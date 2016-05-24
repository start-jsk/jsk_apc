#!/usr/bin/env roseus
(require :unittest "lib/llib/unittest.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter.l")
(load "package://jsk_2015_05_baxter_apc/euslisp/test/data/bin-cubes.l")

(init-unit-test)

(defun setup ()
  (unless (boundp '*baxter*)
    (setq *baxter* (instance jsk_2016_01_baxter_apc::baxter-robot :init)))
  (objects (list *baxter*))
  (send *baxter* :angle-vector #f(0.0 97.4707 -2.39502 -94.5483 134.67 91.4062 8.70117 0.0 -5.40301 55.9257 117.063 72.4793 -62.7243 -7.73117 -87.322 90.0))
  )

(defun check-result (av1 av2)
  (let ()
    (when (and av1 av2)
      (format t "av1 = ~A~%" av1)
      (format t "av2 = ~A~%" av2)
      (format t "diff = ~A (~A)~%" (v- av2 av1) (norm (v- av2 av1)))
      (do ((i 0 (incf i 0.1)))
          ((>= i 1))
          (send *baxter* :angle-vector (midpoint i av1 av2))
          (send *irtviewer* :draw-objects)
          (print i))
      )))
  
(deftest issue-1474 () ;; https://github.com/start-jsk/jsk_apc/issues/1470#issuecomment-220518802
  (let (av1 av2)
    (setup)
    (setq av1 (send *baxter* :angle-vector))
    (send *baxter* :rarm :move-end-pos #f(0 0 -50) :local)
    (setq av2 (send *baxter* :angle-vector)) ;; #f(0.0 97.4707 -2.39502 -94.5483 134.67 91.4062 8.70117 0.0 -8.22871 40.7953 92.2281 64.6958 -80.3615 -21.3969 -46.1696 90.0)
    (check-result av1 av2)
    (assert (and av1 av2))
    (assert (< (norm (v- av1 av2)) 10))
    ))

;; check why normal ik failed
(deftest check-normal-ik-1 () ;; this uses exactry the same parameter with baxter-robot :inverse-kinematics in baxtereus/baxter-util.l
  (let (av1 av2)
    (setup)
    (setq av1 (send *baxter* :angle-vector))
    (send *irtviewer* :draw-objects)
    (setq av2 (send-message *baxter* robot-model :inverse-kinematics
                            (send (send *baxter* :rarm :end-coords :copy-worldcoords) :translate #f(0 0 -50) :local)
                            :additional-weight-list (mapcar #'(lambda (jn) (list (send *baxter* jn :child-link) 0))
                                                            '(:right_gripper_vacuum_pad_joint))
                            :move-target (send *baxter* :rarm :end-coords)
                            :link-list (send *baxter* :link-list (send *baxter* :rarm :end-coords :parent) (send *baxter* :rarm :root-link))
                            :avoid-collision-distance 5
                            :debug-view nil ;; set true for debugging
                            ))
    (check-result av1 av2)
    (assert (and av1 av2)) ;; this is expected to fail without new irteus
    ))

(deftest check-normal-ik-2 () ;; this may fix the problem
  (let (av1 av2)
    (setup)
    (setq av1 (send *baxter* :angle-vector))
    (send *irtviewer* :draw-objects)
    (setq av2 (send-message *baxter* robot-model :inverse-kinematics
                            (send (send *baxter* :rarm :end-coords :copy-worldcoords) :translate #f(0 0 -50) :local)
                            :move-target (send *baxter* :rarm :end-coords)
                            ;;:link-list (send *baxter* :link-list (send *baxter* :rarm :end-coords :parent) (send *baxter* :rarm :root-link))
                            :link-list (send *baxter* :link-list (send *baxter* :right_gripper_vacuum_pad_joint :parent-link) (send *baxter* :rarm :root-link))
                            :avoid-collision-distance 5
                            :debug-view nil ;; set true for debugging
                            ))
    (check-result av1 av2)
    (assert (and av1 av2))
    ))

(run-all-tests)
(exit)
