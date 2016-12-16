;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/main.l")


;; ---------------
;; CONSTANT VALUES
;; ---------------
(setq *bin* :c)
(setq *arm* :rarm)
(setq *use-kinect* nil)


(defun demo-init ()
  (apc-init)
  (setq now (ros::time-now))
  (send *ri* :recognize-bin-boxes :stamp now))


(defun place-object ()
  (send *ri* :move-arm-body->order-bin *arm* :no-gripper-controller)
  (send *ri* :gripper-servo-off *arm*)  ;; power off gripper servo not to overload it
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp *arm*)
  (send *ri* :spin-off-by-wrist *arm* :times 20)
  (send *ri* :gripper-servo-on *arm*)
  (send *ri* :wait-interpolation)
  (send *baxter* :fold-pose-back *arm*)
  (send *ri* :send-av)
  (send *ri* :wait-interpolation)
  )


(defun solidity-main ()
  (ros::set-param
    (format nil "~a_hand/target_bin"
            (send *ri* :arm-symbol2str *arm*))
    (symbol2str *bin*))

  (cond ((not *use-kinect*)
         (send *ri* :move-arm-body->bin-overlook-pose *arm* *bin*)
         (send *ri* :wait-interpolation)
         (unix::sleep 6)
         (ros::set-param
           (format nil "~a_collect_sib_data/ready_to_save" (send *ri* :arm-symbol2str *arm*))
           t)
         (unix::sleep 2)
         (send *ri* :recognize-objects-segmentation-in-bin *arm* *bin* :stamp (ros::time-now))
         )
        (t
          (send *ri* :recognize-objects-segmentation-in-bin *arm* *bin* :stamp (ros::time-now))
          (send *ri* :move-arm-body->bin *arm* *bin*)
          (send *ri* :wait-interpolation)
          )
        )

  (send *ri* :pick-object *arm* *bin* :n-trial 1 :n-trial-same-pos 1)

  (send *baxter* :avoid-shelf-pose *arm* *bin*)
  (send *ri* :send-av 3000 :no-gripper-controller)
  (send *ri* :wait-interpolation)

  (send *baxter* :fold-to-keep-object *arm*)
  (send *ri* :send-av 3000 :no-gripper-controller)
  (send *ri* :wait-interpolation)

  (place-object)
  )
