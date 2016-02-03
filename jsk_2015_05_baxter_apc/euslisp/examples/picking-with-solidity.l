;; vim: set ft=lisp:

(require "package://jsk_2015_05_baxter_apc/euslisp/main.l")


;; ---------------
;; CONSTANT VALUES
;; ---------------
(setq *bin* :f)
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

  ;; (send *ri* :recognize-objects-in-bin *bin* :stamp now)

  (ros::ros-info "Getting solidity rag merge result")
  (send *ri* :recognize-grasp-coords-list *bin* :stamp now)

  (send *ri* :move-arm-body->bin *arm* *bin*)
  (send *ri* :wait-interpolation)

  (send *ri* :ik->bin-entrance *arm* *bin* :offset #f(-50 0 50))
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  ;; (send *ri* :try-to-pick-object *arm* *bin* :offset #f(0 0 100))
  (send *ri* :try-to-pick-object-solidity *arm* *bin* :offset #f(0 0 100))
  (send *ri* :start-grasp *arm*)
  (send *ri* :wait-interpolation)

  (unless (send *ri* :graspingp *arm*)
    (send *baxter* *arm* :move-end-pos #f(0 0 -50) :world)
    (send *ri* :send-av)
    (send *ri* :wait-interpolation)
    (send *baxter* *arm* :move-end-pos #f(0 0 50) :world)
    )

  (send *baxter* *arm* :move-end-pos #f(0 0 100) :world)
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  (send *ri* :ik->bin-entrance *arm* *bin* :offset #f(-200 0 50))
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  (send *baxter* :avoid-shelf-pose *arm* *bin*)
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  (send *baxter* :fold-to-keep-object *arm*)
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  (place-object)
  )
