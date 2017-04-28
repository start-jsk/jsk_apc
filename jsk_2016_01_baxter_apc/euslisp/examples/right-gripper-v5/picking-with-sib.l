;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/main-rgv5.l")


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
  (if (eq *arm* :rarm)
    (progn
      ;; gripper-v5
      (send *ri* :move-hand *arm*
            (send *baxter* :hand-grasp-pre-pose *arm* (send *ri* :get-val '_grasp-style)) 1500)
      (send *baxter* :slide-gripper *arm* 100 :relative nil)
      (send *ri* :send-av 1000)
      (send *ri* :wait-interpolation)
      (send *baxter* :slide-gripper *arm* 0 :relative nil)
      (send *ri* :send-av 1000)
      (send *ri* :wait-interpolation)
      (send *ri* :move-hand *arm*
            (send *baxter* :hand-grasp-pre-pose *arm* :opposed) 1500)
      (send *ri* :spin-off-by-wrist *arm* :times 5)
      )
    (send *ri* :spin-off-by-wrist arm :times 20)
    )
  (send *ri* :gripper-servo-on *arm*)
  (send *ri* :wait-interpolation)
  (send *baxter* :fold-pose-back *arm*)
  (send *ri* :send-av)
  (send *ri* :wait-interpolation)
  )


(defun solidity-main ()
  ;; set target
  (let ((bins (list :a :b :c :d :e :f :g :h :i :j :k :l))
        (label-names (ros::get-param (format nil "/~a_hand_camera/label_names" (arm2str *arm*))))
        order target-obj bin-contents)
    (setq order (send *ri* :get-certain-work-order *arm* *bin*))
    (if (null order)
      (progn
        ;; work order of *bin* doesn't exist
        (ros::ros-warn "[solidity-main] could not find work order of ~a" *bin*)
        (return-from solidity-main nil)
        )
      (progn
        (ros::ros-warn "work-order: ~a" order)
        ;; get target param
        (setq target-obj (send order :object))
        (setq bin-contents (send *ri* :get-bin-contents *bin*))
        (ros::set-dynparam
          (format nil "/~a_hand_camera/bbox_array_to_bbox" (arm2str *arm*))
          (cons "index" (position *bin* bins)))
        (ros::set-dynparam
          (format nil "/~a_hand_camera/label_to_mask" (arm2str *arm*))
          (cons "label_value" (position target-obj label-names :test #'string=)))
        (send *ri* :set-object-segmentation-candidates
              :arm *arm*
              :candidates (mapcar #'(lambda (x) (position x label-names :test #'string=))
                                  (append (list "background") bin-contents)))
        (if (send *ri* :check-bin-exist *bin*)
          (progn
            (ros::set-param
              (format nil "~a_hand/target_bin" (arm2str *arm*))
              (symbol2str *bin*))
            ;; logging
            (ros::ros-info-blue "[solidity-main] target-bin: ~a, target-obj: ~a" *bin* target-obj)
            )
          (progn
            (ros::ros-warn "[solidity-main] could not find bin box: ~a" *bin*)
            (return-from solidity-main nil)
            )
          )
        )
      )
    )

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
