;; This example works with gravity compensation without gripper
(require "package://jsk_2016_01_baxter_apc/euslisp/main-rgv5.l")

;; ---------------
;; CONSTANT VALUES
;; ---------------
(setq *bin* :c)
(setq *arm* :rarm)

(defun demo-init ()
  (apc-init :default-controller nil)
  (setq now (ros::time-now))
  (send *ri* :recognize-bin-boxes :stamp now))


(defun place-object (gripper-angle gripper-l grasp-style)
  ;(send *ri* :move-arm-body->order-bin *arm*)
  (let (avs)
    (pushback (progn
                (send *baxter* :fold-to-keep-object *arm*)
                (send *baxter* :rotate-gripper *arm* gripper-angle :relative nil)
                (send *baxter* :slide-gripper *arm* gripper-l :relative nil)) avs)
    (pushback (progn
                (send *baxter* :place-object-pose *arm*)
                (send *baxter* :rotate-gripper *arm* gripper-angle :relative nil)
                (send *baxter* :slide-gripper *arm* gripper-l :relative nil)) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 4.0)
    )
  (send *ri* :gripper-servo-off *arm*)  ;; power off gripper servo not to overload it
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp *arm*)
  (send *baxter* :hand-grasp-pre-pose *arm* grasp-style)
  (send *ri* :angle-vector
        (send *baxter* :rotate-gripper *arm* 0 :relative nil)
        1500)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand *arm* :angle-vector) 1500)
  (send *ri* :spin-off-by-wrist *arm* :times 20)
  (send *ri* :gripper-servo-on *arm*)
  (send *ri* :wait-interpolation)
  (send *baxter* :fold-pose-back *arm*)
  (send *ri* :send-av)
  (send *ri* :wait-interpolation)
  (send *baxter* :hand-grasp-pre-pose *arm* :opposed)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand *arm* :angle-vector) 1500)
  )


(defun drag-out-bottom-object ()
  (ros::set-param
    (format nil "~a_hand/target_bin"
            (send *ri* :arm-symbol2str *arm*))
    (symbol2str *bin*))

  (send *baxter* :hand-grasp-pre-pose *arm* :opposed)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand *arm* :angle-vector) 1500)
  (unix::sleep 3)
  (send *ri* :move-arm-body->bin *arm* *bin*)
  (send *ri* :wait-interpolation)

  (let (avs ec)
    ;; move arm to bin entrance
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -150 0 20)
                    :gripper-angle -90) avs)
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -30 0 20)
                    :gripper-angle -90) avs)
    ;; slide gripper and prepare fingers
    (setq ec (send *baxter* *arm* :end-coords :copy-worldcoords))
    (send *baxter* :slide-gripper *arm* 124 :relative nil)
    (pushback (send *baxter* *arm* :inverse-kinematics ec) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (send *baxter* :hand-grasp-pre-pose *arm* :cylindrical)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand *arm* :angle-vector) 1500)
    ;; approach to object
    (pushback (send *baxter* *arm* :move-end-pos #f(50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(20 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 -50) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 -50) :world) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    ;; start grasp
    (send *ri* :start-grasp *arm*)
    (unix::sleep 3)
    ;; drag object
    ;(pushback (send *baxter* *arm* :move-end-pos #f(0 0 50) :world) avs)
    ;(pushback (send *baxter* *arm* :move-end-pos #f(0 0 30) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-30 0 0) :world) avs)
    ;; grasp object by hand
    (setq ec (send *baxter* *arm* :end-coords :copy-worldcoords))
    (send *baxter* :slide-gripper *arm* 50 :relative nil)
    (pushback (send *baxter* *arm* :inverse-kinematics ec) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (send *baxter* :hand-grasp-pose *arm* :cylindrical)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand *arm* :angle-vector) 1500)
    ;; draw out object
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 50) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 50) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 50) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-20 0 0) :world) avs)
    (pushback (progn
                (send *baxter* :avoid-shelf-pose *arm* *bin*)
                (send *baxter* :rotate-gripper *arm* -90 :relative nil)
                (send *baxter* :slide-gripper *arm* 50 :relative nil)) avs)
    (pushback (progn
                (send *baxter* :fold-to-keep-object *arm*)
                (send *baxter* :rotate-gripper *arm* -90 :relative nil)
                (send *baxter* :slide-gripper *arm* 50 :relative nil)) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    )

  (place-object -90 50 :cylindrical)
  )

(defun drag-out-middle-object ()
  (ros::set-param
    (format nil "~a_hand/target_bin"
            (send *ri* :arm-symbol2str *arm*))
    (symbol2str *bin*))

  (send *baxter* :hand-grasp-pre-pose *arm* :opposed)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand *arm* :angle-vector) 1500)
  (unix::sleep 3)
  (send *ri* :move-arm-body->bin *arm* *bin*)
  (send *ri* :wait-interpolation)

  (let (avs ec)
    ;; move arm to bin entrance
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -150 0 20)
                    :gripper-angle 90) avs)
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -30 0 20)
                    :gripper-angle 0) avs)
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -20 70 -30)
                    :gripper-angle 0) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (pushback (send *baxter* :slide-gripper *arm* 60 :relative nil) avs)
    (pushback (send *baxter* :slide-gripper *arm* 124 :relative nil) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 2.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    ;; start grasp
    (send *ri* :start-grasp *arm*)
    (unix::sleep 3)
    ;; approach and drag object by sliding gripper
    (send *baxter* :slide-gripper *arm* 60 :relative nil)
    (pushback (send *baxter* :angle-vector) avs)
    (send *baxter* :slide-gripper *arm* 0 :relative nil)
    (pushback (send *baxter* :angle-vector) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 3.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (unix::sleep 1)
    (pushback (send *baxter* *arm* :move-end-pos #f(-70 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(50 0 0) :world) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 6.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    ;; grasp object by hand
    (send *baxter* :hand-grasp-pose *arm* :opposed)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand *arm* :angle-vector) 1500)
    ;; draw out object
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 50) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 10) :world) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 6.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (setq ec (send *baxter* *arm* :end-coords :copy-worldcoords))
    (send ec :rotate 0.2 :y :world)
    (pushback (send *baxter* *arm* :inverse-kinematics ec) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (send *baxter* :avoid-shelf-pose *arm* *bin*)
    (pushback (progn
                (send *baxter* *arm* :move-end-pos #f(-150 0 0) :world)
                (send *baxter* :rotate-gripper *arm* 0 :relative nil)) avs)
    (pushback (progn
                (send *baxter* :fold-to-keep-object *arm*)
                (send *baxter* :rotate-gripper *arm* 0 :relative nil)) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    )

  (place-object 0 0 :opposed)
  )

(defun drag-out-narrow-object ()
  (ros::set-param
    (format nil "~a_hand/target_bin"
            (send *ri* :arm-symbol2str *arm*))
    (symbol2str *bin*))

  (send *baxter* :hand-grasp-pre-pose *arm* :opposed)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand *arm* :angle-vector) 1500)
  (unix::sleep 3)
  (send *ri* :move-arm-body->bin *arm* *bin*)
  (send *ri* :wait-interpolation)

  (let (avs ec)
    ;; move arm to bin entrance
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -150 0 20)
                    :gripper-angle 90) avs)
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -30 0 20)
                    :gripper-angle 0) avs)
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -20 110 -40)
                    :gripper-angle 0) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (pushback (send *baxter* :slide-gripper *arm* 60 :relative nil) avs)
    (pushback (send *baxter* :slide-gripper *arm* 124 :relative nil) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 2.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    ;; start grasp
    (send *ri* :start-grasp *arm*)
    (unix::sleep 3)
    ;; approach and drag object by sliding gripper
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 30) :world) avs)
    (send *baxter* :slide-gripper *arm* 60 :relative nil)
    (pushback (send *baxter* :angle-vector) avs)
    (send *baxter* :slide-gripper *arm* 0 :relative nil)
    (pushback (send *baxter* :angle-vector) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 3.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (unix::sleep 1)
    (pushback (send *baxter* *arm* :move-end-pos #f(-20 0 0) :world) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 6.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    ;; grasp object by hand
    (send *baxter* :hand-grasp-pose *arm* :opposed)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand *arm* :angle-vector) 1500)
    ;; draw out object
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 50) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(0 0 10) :world) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 6.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (setq ec (send *baxter* *arm* :end-coords :copy-worldcoords))
    (send ec :rotate 0.2 :y :world)
    (pushback (send *baxter* *arm* :inverse-kinematics ec) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (send *baxter* :avoid-shelf-pose *arm* *bin*)
    (pushback (progn
                (send *baxter* *arm* :move-end-pos #f(-150 0 0) :world)
                (send *baxter* :rotate-gripper *arm* 0 :relative nil)) avs)
    (pushback (progn
                (send *baxter* :fold-to-keep-object *arm*)
                (send *baxter* :rotate-gripper *arm* 0 :relative nil)) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    )

  (place-object 0 0 :opposed)
  )

(defun drag-out-pencil-case()
  (ros::set-param
    (format nil "~a_hand/target_bin"
            (send *ri* :arm-symbol2str *arm*))
    (symbol2str *bin*))

  (send *baxter* :hand-grasp-pre-pose *arm* :opposed)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand *arm* :angle-vector) 1500)
  (unix::sleep 3)
  (send *ri* :move-arm-body->bin *arm* *bin*)
  (send *ri* :wait-interpolation)

  (let (avs ec)
    ;; move arm to bin entrance
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -150 0 20)
                    :gripper-angle 90) avs)
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -30 0 20)
                    :gripper-angle 0) avs)
    (pushback (send *ri* :ik->bin-entrance *arm* *bin*
                    :offset (float-vector -20 0 10)
                    :gripper-angle 0) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (setq ec (send *baxter* *arm* :end-coords :copy-worldcoords))
    (send ec :rotate -1.57 :x :world)
    (pushback (send *baxter* *arm* :inverse-kinematics ec) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(50 0 0) :world) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    (pushback (send *baxter* :slide-gripper *arm* 60 :relative nil) avs)
    (pushback (send *baxter* :slide-gripper *arm* 124 :relative nil) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 2.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    ;; prepare fingers
    (send *baxter* :hand-grasp-pre-pose *arm* :cylindrical)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand *arm* :angle-vector) 1500)
    (unix::sleep 3)
    ;; approach and drag object by sliding gripper
    (pushback (send *baxter* *arm* :move-end-pos #f(0 40 0) :world) avs)
    (pushback (send *baxter* :rotate-gripper *arm* -45 :relative nil) avs)
    (send *baxter* :slide-gripper *arm* 80 :relative nil)
    (pushback (send *baxter* :angle-vector) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 3.0)
    (setq avs nil)
    (send *ri* :wait-interpolation)
    ;; grasp object by hand
    (send *baxter* :hand-grasp-pose *arm* :cylindrical)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand *arm* :angle-vector) 1500)
    (unix::sleep 3)
    ;; draw out object
    ;(pushback (send *baxter* *arm* :move-end-pos #f(0 0 50) :world) avs)
    ;(pushback (send *baxter* *arm* :move-end-pos #f(0 0 10) :world) avs)
    ;(send *ri* :angle-vector-sequence avs :fast nil 0 :scale 6.0)
    ;(setq avs nil)
    ;(send *ri* :wait-interpolation)
    (pushback (send *baxter* *arm* :move-end-pos #f(-20 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-30 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (pushback (send *baxter* *arm* :move-end-pos #f(-50 0 0) :world) avs)
    (send *baxter* :avoid-shelf-pose *arm* *bin*)
    (pushback (progn
                (send *baxter* *arm* :move-end-pos #f(-150 0 0) :world)
                (send *baxter* :rotate-gripper *arm* -45 :relative nil)
                (send *baxter* :slide-gripper *arm* 80 :relative nil)) avs)
    (pushback (progn
                (send *baxter* :fold-to-keep-object *arm*)
                (send *baxter* :rotate-gripper *arm* -45 :relative nil)
                (send *baxter* :slide-gripper *arm* 80 :relative nil)) avs)
    (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 5.0)
    (send *ri* :wait-interpolation)
    )

  (place-object -45 80 :opposed)
  )
