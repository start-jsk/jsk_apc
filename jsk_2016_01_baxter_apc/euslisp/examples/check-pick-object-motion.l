;; For gripper-v5
;; CAUTION: This script works only in simulation
#!/usr/bin/env roseus

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")

(ros::roseus "check_pick_object_motion")

(setq arm :rarm)
(if (eq arm :larm)
  (setq bin-list (list :a :b :d :e :g :j))
  (setq bin-list (list :c :f :h :i :k :l))
  )

(jsk_2016_01_baxter_apc::baxter-init)
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
        (list (send *ri* :ik->bin-entrance arm bin
                    :offset (float-vector -150 0 0)
                    ))
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
      (send *ri* :ik->bin-entrance arm bin
            :offset (float-vector -150 0 0)
            :gripper-angle gripper-angle
            :rotation-axis t)
      (setq ec (send *baxter* arm :end-coords :copy-worldcoords))
      (send ec :rotate (* sign pi/2) :x :world)
      ;; initial pose for ik
      (send *ri* :ik->bin-entrance arm bin
            :offset (float-vector -150 0 0)
            :rotation-axis t)
      (send *baxter* :rotate-gripper arm gripper-angle :relative nil)
      ;; ik to target coords
      (pushback (send *ri* :rotate-wrist-ik arm ec) avs)
      (pushback (send *baxter* arm :move-end-pos #f(150 0 0) :world) avs)
      (pushback (send *baxter* arm :move-end-pos #f(150 0 0) :world) avs)
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
