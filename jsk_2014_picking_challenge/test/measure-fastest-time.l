#!/usr/bin/env roseus
(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")
(load "package://jsk_2014_picking_challenge/euslisp/gripper-control.l")
(load "package://jsk_2014_picking_challenge/euslisp/arm-motion.l")
(load "package://jsk_2014_picking_challenge/euslisp/main.l")


(defun test (l/r)
  (let (targets '(:a :b :c :d :e :f :g :h :i :j :k :l))
    (dolist (t targets)
      (send *ri* :angle-vector-sequence (fold-pose l/r t) :fast :default-controller 0 :scale 2.5)
      (send *ri* :wait-interpolation)
      (move-arm-to-target-bin l/r t)
      (send *ri* :angle-vector-sequence (pick-object l/r) :fast :default-controller 0 :scale 2.5)
      ))
  )
