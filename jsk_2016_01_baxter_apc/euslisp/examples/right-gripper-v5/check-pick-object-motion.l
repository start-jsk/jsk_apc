;; For gripper-v5
;; CAUTION: This script works only in simulation
#!/usr/bin/env roseus

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxterrgv5-interface.l")

(ros::roseus "check_pick_object_motion")

(setq arm :rarm)
(if (eq arm :larm)
  (setq bin-list (list :a :b :d :e :g :j))
  (setq bin-list (list :c :f :h :i :k :l))
  )

(jsk_2016_01_baxter_apc::baxterrgv5-init)
;; check if it's simulation mode
(unless (send *ri* :simulation-modep)
  (ros::ros-fatal "[~a] Please run this script in simulation" (ros::get-name))
  (exit)
  )

(send *ri* :angle-vector
      (send *baxter* :fold-pose-back arm)
      5000)
(send *ri* :wait-interpolation)

(send *ri* :recognize-bin-boxes)

(dolist (bin bin-list)
  (ros::ros-info-green "[~a] Moving arm ~a to bin ~a" (ros::get-name) arm bin)
  (unix::sleep 2)
  (send *ri* :move-arm-body->bin-overlook-pose arm bin)
  (send *ri* :wait-interpolation)
  (send *ri* :angle-vector-sequence
        (list
          (send *ri* :ik->bin-entrance arm bin
                :offset (float-vector -150 0 20))
          (send *ri* :ik->bin-entrance arm bin
                :offset (float-vector -150 0 20)
                :rotation-axis t)
          )
        :fast nil 0 :scale 5.0)
  (send *ri* :wait-interpolation)
  ;; move to target
  (dolist (gripper-angle '(90 -90))
    (dolist (sign '(0 -1 1))
      (ros::ros-info-green "[~a] arm: ~a, bin: ~a, gripper angle: ~a, sign: ~a"
                     (ros::get-name) arm bin gripper-angle sign)
      (unix::sleep 2)
      (setq avs (list (send *baxter* :angle-vector)))
      ;; get target coords
      (setq bin-cube-base (send *ri* :cube->cube-parallel-to-coords
                                (send *ri* :bbox->cube
                                      (gethash bin (send *ri* :get-val '_bin-boxes)))
                                :included-in-original-cube))
      (setq bin-center (make-cascoords))
      (send bin-center :newcoords
            (send bin-cube-base :copy-worldcoords))
      (setq end-coords (make-cascoords))
      (send end-coords :newcoords
            (send bin-center :copy-worldcoords))
      (send end-coords :translate
            (v- #f(-150 0 60) (float-vector (/ (x-of-cube bin-cube-base) 2) 0 0))
            :world)
      (send bin-center :assoc end-coords)
      (send bin-center :rotate (* sign pi/2) :x :local)
      (if (= gripper-angle -90)
        (send end-coords :rotate pi :z :local))
      ;; initial pose for ik
      (send *ri* :ik->bin-entrance arm bin
            :offset (float-vector -150 0 60)
            :rotation-axis t)
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      ;; ik to target coords
      ;(setq ik-poses-orig (send *baxter* :ik-prepared-poses))
      ;(send *baxter* :set-ik-prepared-poses nil)
      (pushback (send *ri* :rotate-wrist-ik arm end-coords :rthre (deg2rad 1)) avs)
      (send end-coords :translate #f(120 0 0) :world)
      (pushback (send *ri* :rotate-wrist-ik arm end-coords :rthre (deg2rad 1)) avs)
      (send end-coords :translate #f(0 0 -60) :local)
      (send end-coords :translate #f(30 0 0) :world)
      (if (send *baxter* arm :inverse-kinematics end-coords)
        (pushback (send *baxter* :angle-vector) avs)
        (ros::ros-error
          "[~a] There is IK unsolvable region in bin ~a when arm: ~a, gripper angle: ~a, sign: ~a"
          (ros::get-name) bin arm gripper-angle sign)
        )
      (if (send *baxter* arm :move-end-pos #f(150 0 0) :world)
        (pushback (send *baxter* :angle-vector) avs)
        (ros::ros-error
          "[~a] There is IK unsolvable region in bin ~a when arm: ~a, gripper angle: ~a, sign: ~a"
          (ros::get-name) bin arm gripper-angle sign)
        )
      ;(send *baxter* :set-ik-prepared-poses ik-poses-orig)
      (send *ri* :angle-vector-sequence avs
            :fast nil 0 :scale 5.0)
      (send *ri* :wait-interpolation)
      (unix::sleep 2)
      (send *baxter* :angle-vector (car avs))
      (send *ri* :angle-vector-sequence (reverse avs)
            :fast nil 0 :scale 5.0)
      (send *ri* :wait-interpolation)
      )
    )
  (ros::ros-info-green "[~a] arm: ~a finish checking of bin ~a" (ros::get-name) arm bin)

  (send *ri* :angle-vector
    (send *baxter* :avoid-shelf-pose arm bin)
    5000)
  (send *ri* :wait-interpolation)

  (send *ri* :angle-vector
        (send *baxter* :fold-to-keep-object arm)
        5000)
  (send *ri* :wait-interpolation)
  )
(exit)
