#!/usr/bin/env roseus

(require "package://baxtergv6_apc2016/euslisp/lib/baxter-interface.l")

(setq *init-av*
      #f(9.51416 11.3599 -80.0244 -53.042 103.821 8.30566 69.5874 80.3979 1.1781 -0.178926
         2.17944 -43.9569 10.8806 46.7142 -4.91307 87.4745 -42.1617 0.957204 1.25249))

(defun demo-init ()
  (jsk_arc2017_baxter::baxter-init :type :rarm-controller :moveit nil)
  ;; initialize fingers
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pose :rarm :cylindrical :angle 40) 1000)
  (send *ri* :angle-vector (send *ri* :state :potentio-vector) 1000)
  (send *ri* :angle-vector (send *baxter* :angle-vector *init-av*) 10000)
  (send *ri* :wait-interpolation)
  (unix::sleep 1)
  (send *ri* :calib-proximity-threshold :rarm)
  ;(send *ri* :calib-pressure-threshold :rarm)
  (objects (list *baxter*)))

(defun down-with-sensor-stop (&key (down-z -110))
  (send *ri* :angle-vector
        (send *baxter* :rarm :move-end-pos (float-vector 0 0 down-z) :world)
        6000 (send *ri* :get-arm-controller :rarm) 0)
  (send *ri* :wait-interpolation-until :rarm
        :finger-flexion :finger-loaded :finger-proximity)
  (unix::sleep 1)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (send *ri* :angle-vector (send *baxter* :slide-gripper :rarm 60 :relative nil) 1000)
  (send *ri* :wait-interpolation)
  (send *ri* :move-hand :rarm
        (send *baxter* :hand-grasp-pose :rarm :cylindrical) 1000)
  (send *ri* :angle-vector
        (send *baxter* :rarm :move-end-pos (float-vector 0 0 (- down-z)) :world)
        2000 (send *ri* :get-arm-controller :rarm) 0)
  )
