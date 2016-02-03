;; vim: set ft=lisp:

(require "package://jsk_2015_05_baxter_apc/euslisp/main.l")


;; ---------------
;; CONSTANT VALUES
;; ---------------
(setq *bin* :e)
(setq *arm* :rarm)


(defun demo-init ()
  (apc-init)
  (setq now (ros::time-now))
  (send *ri* :recognize-bin-boxes :stamp now))


(defun place-object ()
  (send *ri* :move-arm-body->order-bin *arm*)
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp *arm*)
  (send *ri* :spin-off-by-wrist *arm* :times 20)
  (send *ri* :wait-interpolation)
  (send *baxter* :fold-pose-back *arm*)
  (send *ri* :send-av)
  (send *ri* :wait-interpolation)
  )


(defun solidity-main ()
  (setq now (ros::time-now))

  (send *ri* :recognize-objects-in-bin *bin* :stamp now)

  ; (ros::ros-info "Getting solidity rag merge result")
  ; (send *ri* :recognize-grasp-coords-list *bin* :stamp now)

  (send *ri* :move-arm-body->bin *arm* *bin*)
  (send *ri* :wait-interpolation)

  (send *ri* :pick-object *arm* *bin* :n-trial 1 :n-trial-same-pos 1)

  (send *baxter* :avoid-shelf-pose *arm* *bin*)
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  (send *baxter* :fold-to-keep-object *arm*)
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  (place-object)
  )
