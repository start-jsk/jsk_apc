#!/usr/bin/env roseus

(require "package://jsk_arc2017_baxter/euslisp/lib/baxter-interface.l")

(setq *init-av*
      #f(2.92236 11.4258 -80.0244 -53.064 103.733 8.28369 69.5874 80.3979 0.883573 0.0
         -3.88756 -43.5338 16.0149 65.4753 -12.2104 67.7277 -39.7833 0.368155 0.894632))

(setq *av-before-place*
      #f(2.90039 11.4038 -80.0244 -53.064 103.645 8.30566 69.6094 80.3979 1.03084 -0.178926
         -44.6924 -39.5288 14.48 92.9224 86.6382 78.5522 -55.1733 0.257709 1.25249))

(defun demo-init ()
  (jsk_arc2017_baxter::baxter-init :type :default-controller :moveit nil)
  (send *ri* :angle-vector (send *baxter* :angle-vector *init-av*) 10000)
  (send *ri* :wait-interpolation)
  ;; initialize fingers
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pre-pose :rarm :cylindrical) 1000)
  ;(send *ri* :calib-pressure-threshold :rarm)
  (objects (list *baxter*)))

(defun fixed-pick-book (&key (down-z -60))
  (send *ri* :angle-vector
        (send *baxter* :rarm :move-end-pos (float-vector 0 0 down-z) :world) 3000)
  (send *ri* :wait-interpolation)
  (send *ri* :start-grasp :rarm)
  (unix::sleep 2)
  (send *ri* :angle-vector (send *baxter* :rarm :move-end-pos #f(0 0 300) :world) 5000)
  (send *ri* :wait-interpolation)
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pose :rarm :cylindrical) 1000)
  (send *ri* :angle-vector (send *baxter* :angle-vector *av-before-place*) 5000)
  (send *ri* :wait-interpolation)
  (send *ri* :angle-vector (send *baxter* :rarm :move-end-pos #f(160 0 0) :world) 3000)
  (send *ri* :wait-interpolation)
  (send *ri* :angle-vector (send *baxter* :rarm :move-end-pos #f(0 100 0) :world) 3000)
  (send *ri* :wait-interpolation)
  (send *ri* :angle-vector (send *baxter* :rarm :move-end-pos #f(0 0 -250) :world) 3000)
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp :rarm)
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pre-pose :rarm :cylindrical) 1000)
  (unix::sleep 1)
  (send *ri* :angle-vector (send *baxter* :rarm :move-end-pos #f(0 0 250) :world) 3000)
  (send *ri* :wait-interpolation)
  (send *ri* :angle-vector (send *baxter* :rarm :move-end-pos #f(-160 0 0) :world) 3000)
  (send *ri* :wait-interpolation)
  )
