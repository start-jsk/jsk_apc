(require "package://jsk_arc2017_baxter/euslisp/lib/baxterlgv8-interface.l")


;; ---------------
;; CONSTANT VALUES
;; ---------------
(setq *arm* :larm)


(defun demo-init
  (&key (type :lgripper-controller))
  (jsk_arc2017_baxter::baxterlgv8-init :type type)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand-grasp-pre-pose *arm* :cylindrical) 1500)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500)
  (send *baxter* :rotate-gripper *arm* 0 :relative nil)
  (send *baxter* :slide-gripper *arm* 0 :relative nil)
  (send *ri* :angle-vector (send *baxter* :angle-vector))
  (send *ri* :wait-interpolation)
  (objects (list *baxter*))
  t)

(defun gv8-movement (&key (move-fingers t))
  (when move-fingers
    ;; test of abduction/adduction
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pre-pose *arm* :cylindrical) 1500)
    (unix::sleep 2)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500)
    (unix::sleep 2)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pose *arm* :opposed :angle 0) 1500)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pose *arm* :cylindrical :angle 0) 1500)
    (unix::sleep 2)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pose *arm* :opposed :angle 0) 1500)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500)
    (unix::sleep 2)
    ;; test of closing fingers
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pose *arm* :opposed :angle 80) 1500)
    (unix::sleep 1)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500)
    (unix::sleep 1)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pose *arm* :opposed :angle 100) 1500)
    (unix::sleep 1)
    (send *ri* :move-hand *arm*
          (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500))
  ;; test of prismatic joint
  (send *baxter* :slide-gripper *arm* 206 :relative nil)
  (send *ri* :angle-vector (send *baxter* :angle-vector) :fast nil 0)
  (send *ri* :wait-interpolation)
  (unix::sleep 4)
  ;; test of pad joint
  (send *baxter* :rotate-gripper *arm* 90 :relative nil)
  (send *ri* :angle-vector (send *baxter* :angle-vector) :fast nil 0)
  (send *ri* :wait-interpolation)
  (send *baxter* :rotate-gripper *arm* -90 :relative nil)
  (send *ri* :angle-vector (send *baxter* :angle-vector) :fast nil 0)
  (send *ri* :wait-interpolation)
  (unix::sleep 1)
  (send *baxter* :rotate-gripper *arm* 0 :relative nil)
  (send *ri* :angle-vector (send *baxter* :angle-vector) :fast nil 0)
  (send *ri* :wait-interpolation)
  ;; test of prismatic joint
  (send *baxter* :slide-gripper *arm* 0 :relative nil)
  (send *ri* :angle-vector (send *baxter* :angle-vector) :fast nil 0)
  (send *ri* :wait-interpolation)
  )
