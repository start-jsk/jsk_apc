(require "package://jsk_2016_01_baxter_apc/euslisp/main-rgv5.l")


;; ---------------
;; CONSTANT VALUES
;; ---------------
(setq *arm* :rarm)


(defun demo-init
  (&key (ctype :rgripper-controller))
  (jsk_2016_01_baxter_apc::baxterrgv5-init :ctype ctype)
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

(defun solidity-main ()
  ;; test of abduction/adduction
  (send *ri* :move-hand *arm*
        (send *baxter* :hand-grasp-pre-pose *arm* :cylindrical) 1500)
  (unix::sleep 2)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500)
  (unix::sleep 2)
  ;; test of closing fingers
  (send *ri* :move-hand *arm*
        (send *baxter* :hand-grasp-pose *arm* :opposed :angle 130) 1500)
  (unix::sleep 1)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500)
  (unix::sleep 1)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand-grasp-pose *arm* :opposed :angle 130) 1500)
  (unix::sleep 1)
  (send *ri* :move-hand *arm*
        (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500)
  ;; test of prismatic joint
  (send *baxter* :slide-gripper *arm* 100 :relative nil)
  (send *ri* :angle-vector (send *baxter* :angle-vector) :fast nil 0)
  (send *ri* :wait-interpolation)
  (unix::sleep 3)
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
