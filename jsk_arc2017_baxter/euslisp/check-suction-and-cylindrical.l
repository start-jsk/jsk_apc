#!/usr/bin/env roseus

(require "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")

(ros::load-ros-manifest "jsk_arc2017_baxter")

(unless (boundp '*baxter*) (jsk_arc2017_baxter::arc-init :ctype :rarm-controller))

(defun test ()
  (let (prev-av next-av gripper-x pre-end-coords)
    (setq prev-av (send *baxter* :angle-vector))
    (send *ri* :move-hand :rarm
          (send *baxter* :hand-grasp-pose :rarm :cylindrical :angle 0) 1000)
    ;; :pick-object-with-movable-region
    (setq next-av
          ;; :rotate-gripper :rarm 0
          ;; :slide-gripper :rarm 80
          ;; end-coords #f(600 -700 0)
          (send *baxter* :rarm :angle-vector
                #f(-4.5161 -49.3405 25.7123 47.8614 -16.4674 94.2234 -30.6975 80.0 0.0)))
    (setq gripper-x (send *baxter* :rarm :gripper-x :joint-angle))
    (send *baxter* :angle-vector prev-av)
    ;; First, move prismatic joint to target position
    (send *ri* :angle-vector-raw
          (send *baxter* :slide-gripper :rarm gripper-x :relative nil)
          :fast (send *ri* :get-arm-controller :rarm) 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    ;; Fold fingers to avoid collision
    (send *ri* :move-hand :rarm
          (send *baxter* :hand-grasp-pre-pose :rarm :cylindrical) 1000)
    (send *ri* :move-hand :rarm
          (send *baxter* :hand-grasp-pose :rarm :cylindrical :angle 110) 1000)
    ;; Move whole arm to target pose
    (send *baxter* :rarm :angle-vector next-av)
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000)

    (send *ri* :wait-interpolation)
    ;; :try-to-suction-and-pinch-object
    (send *ri* :move-hand :rarm (send *baxter* :hand-grasp-pose :rarm :cylindrical :angle 110) 1000)

    (send *ri* :start-grasp :rarm)

    ;;(break)

    ;; :rotate-gripper :rarm 0
    ;; :slide-gripper :rarm 80
    ;; end-coords #f(600 -700 -300)
    (send *baxter* :rarm :angle-vector
          #f(-4.48961 -32.8098 21.798 65.4342 -20.9014 61.0237 -26.78 80.0 0.0))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 6000)
    ;; Wait until finger touch
    (send *ri* :wait-interpolation-until :rarm
          :grasp :finger-flexion :finger-loaded :prismatic-loaded)
    (unix::sleep 3)
    (send *baxter* :angle-vector
          (send *ri* :state :potentio-vector :wait-until-update t))
    (setq pre-end-coords (send *baxter* :rarm :end-coords :copy-worldcoords))
    ;; cylindrically grasp object
    (send *ri* :move-hand :rarm
          (send *baxter*
                :hand-grasp-pose :rarm :cylindrical :angle 40) 1000)
    (send *ri* :angle-vector-raw
          (send *baxter* :slide-gripper :rarm 0 :relative nil)
          3000 (send *ri* :get-arm-controller :rarm) 0)
    (send *ri* :wait-interpolation)
    (send *ri* :move-hand :rarm
          (send *baxter*
                :hand-grasp-pose :rarm :cylindrical) 1000 :wait nil)
    (send *ri* :wait-interpolation-until :rarm :finger-flexion :hand)
    (send pre-end-coords :translate #f(0 0 30))
    (send *ri* :angle-vector-raw
          (send *baxter* :rarm :inverse-kinematics pre-end-coords :rotation-axis t :use-gripper nil)
          3000)
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp :rarm :suction)
    (unix::sleep 1)
    (send *ri* :move-hand :rarm
          (send *baxter*
                :hand-grasp-pose :rarm :cylindrical :angle 120) 1000)
    (send *baxter* :rarm :move-end-pos #f(0 0 10))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (send *baxter* :rotate-gripper :rarm -90 :relative nil)
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (send *ri* :start-grasp :rarm)
    (unix::sleep 1)
    (send *baxter* :rotate-gripper :rarm 0 :relative nil)
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000)
    (send *ri* :move-hand :rarm
          (send *baxter*
                :hand-grasp-pose :rarm :cylindrical :angle 180) 4000)
    (unix::sleep 1)
    (send *baxter* :rarm :angle-vector
          #f(-4.5161 -49.3405 25.7123 47.8614 -16.4674 94.2234 -30.6975 80.0 0.0))
    (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000
          (send *ri* :get-arm-controller :rarm :gripper nil) 0)
    (send *ri* :wait-interpolation)
    ;; (unix::sleep 3)
    ;; (send *ri* :stop-grasp :rarm :suction)
    ;; (send *ri* :move-hand :rarm (send *baxter* :hand :rarm :angle-vector #f(90 0)) 1000)))
