#!/usr/bin/env roseus

(ros::roseus "robot_test")

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")

(defun ir2ri (&optional (tm nil))
  (send *ri* :angle-vector (send *baxter* :angle-vector) tm)
  (send *irtviewer* :draw-objects))

(defun ri2ir ()
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *irtviewer* :draw-objects))

(defun test-for-bin (arm bin &key (verify nil))
  (ros::ros-info "test for ~A" bin)
  (move-to-target-bin arm bin)
  (pick-object arm bin)
  (place-object arm))

(ros::advertise "tweet" std_msgs::string 1)

(defun test-challenge (&key
              (bins (list :a :b :c :d :e :f :g :h :i :j :k :l))
              (use-arm (list :larm :rarm))
              (tweet? t)
              (verify? nil))
  (let ((start-time (ros::time-now))
        elapsed-time elapsed-msg arm msg)
    (dolist (bin bins)
      (if (eq (length use-arm) 1) (setq arm (elt use-arm 0))  ;; only use arms in use-arm
        (cond
          ((find bin (list :a :d :g :j)) (setq arm :larm))
          ((find bin (list :c :f :i :l)) (setq arm :rarm))
          (t (setq arm :larm))))
      (test-for-bin arm bin :verify verify?))
    (setq elapsed-time (ros::time- (ros::time-now) start-time))
    (setq elapsed-msg (format nil "It takes ~d min ~d sec" (/ (send elapsed-time :sec) 60) (mod (send elapsed-time :sec) 60)))
    (when tweet? (tweet elapsed-msg))  ;; tweet elapsed time
    ;; return to fold pose and speak
    (send *ri* :angle-vector-sequence (fold-pose-back arm))
    (send *ri* :wait-interpolation)
    (speak-en "Finished picking test." :google t)
    ))

(apc-init)
(warn "~% Commands ~%")
(warn "(test) : Start test~%")
