#!/usr/bin/env roseus
;; vim: set ft=lisp:

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter-interface.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")

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
  (ros::roseus "generate-object-motion-main" :anonymous nil)
  (jsk_2016_01_baxter_apc::baxter-init :ctype :larm-controller)
  (objects (list *baxter*))

  (send *ri* :gripper-servo-on :larm)

  (send *baxter* :fold-pose-back)
  (model2real :scale 8.0)
  (send *ri* :wait-interpolation)
  )

(defun motion-on-tabletop ()
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
    avs)
  )


(defun motion-on-supporting-system ()
  (let (avs)
    (send *baxter* :larm :angle-vector
          #f(86.1987 -7.84424 -1.93359 85.0342 -18.3691 74.8608 174.595 0.0))
    (model2real :scale 8.0)

    ;; (send *baxter* :larm :angle-vector #f(-0.966797 -2.26318 31.6626 62.666 -64.7534 96.8994 171.541 0.0))
    ;; (pushback (send *baxter* :angle-vector) avs)

    (send *baxter* :larm :angle-vector
          #f(30.0366 11.7334 40.2319 16.9849 -77.2119 113.115 151.633 0.0))
    (pushback (send *baxter* :angle-vector) avs)

    (send *baxter* :larm :angle-vector
          #f(-32.3877 -31.8384 56.1841 35.7275 -52.4268 119.971 132.979 0.0))
    (pushback (send *baxter* :angle-vector) avs)

    (send *baxter* :larm :angle-vector
          #f(-53.3716 -36.2769 6.1084 42.5171 -6.21826 101.052 -17.9077 0.0))
    (pushback (send *baxter* :angle-vector) avs)

    (send *baxter* :larm :angle-vector
          #f(-92.0654 -21.1377 8.6792 36.5845 15.4248 113.73 34.4092 0.0))
    (pushback (send *baxter* :angle-vector) avs)
    avs)
  )


(defun main ()
  (let ((object-index 0) (env (ros::get-param "~env" "supporting_system")))
    (ros::ros-info-green "Generating object model on: ~a" env)
    (while (send *ri* :wait-for-user-input-to-start :larm)
      (ros::set-dynparam "/kinfu"
        (cons "save_dir"
              (format nil "~a/.ros/jsk_arc2017_baxter/kinfu_model_generation/object_~a" (unix::getenv "HOME") object-index)))
      ;; larm-overlook-table
      (let (avs)
        (setq avs
              (cond
                ((string= env "supporting_system") (motion-on-supporting-system))
                ((string= env "tabletop") (motion-on-tabletop))
                (t (ros::ros-info-red "Unsupported env: ~a" env))
                )
              )

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
