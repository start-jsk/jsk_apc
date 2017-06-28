#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/util.l")

(defun model2real (&key (wait? t) (scale 16.0))
  (send *ri* :angle-vector-sequence (list (send *baxter* :angle-vector)) :fast nil 0 :scale scale)
  (when wait?
    (send *ri* :wait-interpolation)
    )
  )

(defun real2model (&key (wait? nil))
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send *irtviewer* :redraw)
  )


(defun setup ()
  (jsk_2016_01_baxter_apc::baxter-init :ctype :larm-controller)
  (objects (list *baxter*))

  (send *ri* :gripper-servo-on :larm)

  (send *baxter* :fold-pose-back)
  (model2real :scale 8.0)
  (send *ri* :wait-interpolation)
  )

(defun main ()
  (let ((object-index 0))
    (while (send *ri* :wait-for-user-input-to-start :larm)
      (ros::set-dynparam "/kinfu"
        (cons "save_dir"
              (format nil "~a/.ros/jsk_arc2017_baxter/kinfu_model_generation/object_~a" (unix::getenv "HOME") object-index)))
      ;; larm-overlook-table
      (let (avs)
        (send *baxter* :larm :angle-vector
          #f(97.4927 -29.5312 -1.53809 142.537 81.167 -57.7441 27.7075 0.0)
          )
        (model2real :scale 8.0)

        (send *baxter* :larm :angle-vector
          #f(76.9043 -35.5078 15.1611 67.7197 131.902 -77.4536 -5.64697 0.0)
          )
        (pushback (send *baxter* :angle-vector) avs)

        (send *baxter* :larm :angle-vector
          #f(37.4414 -45.6152 27.356 47.2412 154.072 -86.7261 8.78906 0.0)
          )
        (pushback (send *baxter* :angle-vector) avs)

        (send *baxter* :larm :angle-vector
          #f(2.83447 -60.6885 46.4722 63.7207 156.401 -85.1221 33.9258 0.0)
          )
        (pushback (send *baxter* :angle-vector) avs)

        (ros::service-call "/kinfu/reset" (instance std_srvs::EmptyRequest :init))

        (send *ri* :angle-vector-sequence avs :fast nil 0 :scale 12.0)
        (send *ri* :wait-interpolation)

        (let ((req (instance jsk_gui_msgs::YesNoRequest :init)))
          (ros::ros-info-green "Please select to send save mesh request or not.")
          (setq res (ros::service-call "/rviz/yes_no_button" req))
          (when (send res :yes)
            (ros::ros-info-green "Requesting to save mesh.")
            (ros::service-call "/save_mesh_server/request" (instance std_srvs::EmptyRequest :init))
            (incf object-index)
            ))

        (send *ri* :angle-vector-sequence (reverse avs) :fast nil 0 :scale 5.0)
        (send *ri* :wait-interpolation-smooth 2000)
      )

      (send *baxter* :fold-pose-back)
      (model2real :scale 8.0)
      )
    )
  )
