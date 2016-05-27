;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/main.l")


;; ---------------
;; CONSTANT VALUES
;; ---------------
(setq *bin* :c)
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
  (ros::set-param (format nil "~a_hand/target_bin" (send *ri* :arm-symbol2str *arm*)) (symbol2str *bin*))

  (send *ri* :recognize-objects-in-bin *bin* :stamp (ros::time-now))

  ; (ros::ros-info "Getting solidity rag merge result")
  ; (send *ri* :recognize-grasp-coords-list *bin* :stamp (ros::time-now))

  (send *ri* :move-arm-body->bin-overlook-pose *arm* *bin*)
  (send *ri* :wait-interpolation)
  (unix::sleep 2)
  (send *ri* :recognize-objects-segmentation-in-bin *arm* *bin* :stamp (ros::time-now))

  (send *ri* :pick-object *arm* *bin* :n-trial 1 :n-trial-same-pos 1)

  (send *baxter* :avoid-shelf-pose *arm* *bin*)
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  (send *baxter* :fold-to-keep-object *arm*)
  (send *ri* :send-av 3000)
  (send *ri* :wait-interpolation)

  (place-object)
  )
