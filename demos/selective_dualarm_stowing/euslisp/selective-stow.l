#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")

(setq *target-bin* :b)
(setq *target-object-name* "hanes_tube_socks")

#|
cherokee_easy_tee_shirt
creativity_chenille_stems
expo_eraser
feline_greenies
hanes_tube_socks
ice_cube_tray
kleenex_paper_towels
mark_twain_book
peva_shower_curtain_liner
robots_everywhere
|#
;; SET TARGET

(defun selective-stow-init (&key (calib-pressure t))
  (jsk_2016_01_baxter_apc::baxter-init :ctype :default-controller :moveit t)
  (send *ri* :gripper-servo-on)
  (send *ri* :angle-vector-raw (send *baxter* :fold-pose-back))
  (when calib-pressure
    (send *ri* :calib-pressure-threshold))
  (setq *co* (instance collision-object-publisher :init))
  ;; initialize collision object
  (send *co* :wipe-all)
  (setq *kiva-pod* (make-cube 850 850 2000))
  (setq *kiva-pod-coords* (make-coords :pos (float-vector 1500 0 0)))
  (setq *tote* (make-cube 400 500 300))
  (setq *tote-coords* (make-coords :pos (float-vector 750 0 -450)))
  (setq *target-object* (make-cube 200 200 200))
  (setq *target-object-coords* (make-coords :pos (float-vector 0 0 150)))
  (ros::ros-info "recognizing bin boxes")
  (send *ri* :recognize-bin-boxes :stamp (ros::time-now))
  (ros::ros-info "recognizing order-bin")
  (objects (list *baxter*)))

(defun apc-score (proba)
  (let
    (score)
    (warning-message 1 "drop           : ~a~%" (elt proba 0))
    (warning-message 1 "protrude       : ~a~%" (elt proba 1))
    ; (warning-message 1 "damage         : ~a~%" (elt proba 2))
    (setq score
      (- 10.0
         (+
           (* 10.0 (elt proba 0)) ; drop
           (* 5.0 (elt proba 1))  ; protrude
           ; (* 5.0 (elt proba 2))  ; damage
         )))
    (if (> score 0) score 0)))

(defun evaluate-cls-msg (msg)
  (let
    (proba singlearm-proba dualarm-proba singlearm-score dualarm-score)
    (setq proba (send msg :probabilities))
    (setq singlearm-proba (subseq proba 0 2))
    (setq dualarm-proba (subseq proba 2 4))
    (warning-message 1 "===============================================~%")
    (warning-message 1 "singlearm~%")
    (setq singlearm-score (apc-score singlearm-proba))
    (warning-message 1 "singlearm-score: ~a~%" singlearm-score)
    (warning-message 1 "~%dualarm~%")
    (setq dualarm-score (apc-score dualarm-proba))
    (warning-message 1 "dualarm-score  : ~a~%" dualarm-score)
    (warning-message 1 "===============================================~%")
    (or
      (> (- dualarm-score singlearm-score) 0.5)
      (< singlearm-score 5.0))))

(defun add-kiva-pod-scene ()
  (send *co* :add-object *kiva-pod* :frame-id "base" :relative-pose *kiva-pod-coords* :object-id "kiva_pod"))

(defun add-tote-scene ()
  (send *co* :add-object *tote* :frame-id "base" :relative-pose *tote-coords* :object-id "tote"))

(defun selective-stow (arm is-bimanual)
  (let
    (opposite-arm
      arm-controller
      opposite-arm-controller
      tote-contents
      label-names
      graspingp
      opposite-arm-graspingp
      dropped
      path-constraints
      entrance-x-list
      entrance-x-list-with-opposite-arm
      entrance-x-list-without-opposite-arm
      offset-avoid-bin-top
      offset-avoid-bin-top-exit
      gripper-angle
      opposite-arm-distance
      opposite-arm-wrist-angle
      avs-overlook->tote
      avs-picked->place-bin
      avs-picked->place-bin-with-opposite-arm
      avs-picked->place-bin-without-opposite-arm
      avs-place-bin->exit
      avs-hold-opposite->fold-pose-back
      avs-overlook->fold-pose-back
      cls-msg)

    (ros::ros-info-green (format nil "target object: ~a" *target-object-name*))
    (send *ri* :gripper-servo-on)
    (cond
      ((eq arm :larm)
       (setq opposite-arm :rarm)
       (setq arm-controller :larm-controller)
       (setq opposite-arm-controller :rarm-controller))
      ((eq arm :rarm)
       (setq opposite-arm :larm)
       (setq arm-controller :rarm-controller)
       (setq opposite-arm-controller :larm-controller))
      (t (warning-message 1 "arm: ~A is not proper~%" arm)))
    (setq opposite-arm (if (eq arm :larm) :rarm :larm))
    (send *ri* :recognize-order-bin-box arm :stamp (ros::time-now))
    (send *ri* :get-movable-region-for-tote arm :offset (list 60 80 0))
    (ros::set-param
      (format nil "~a_hand/target_object" (arm2str arm)) *target-object-name*)
    (setq tote-contents (list *target-object-name*))
    ; (setq label-names
    ;       (ros::get-param (format nil "/~a_hand_camera/label_names"
    ;                               (arm2str arm))))
    ; (ros::set-dynparam
    ;   (format nil "/~a_hand_camera/label_to_mask" (arm2str arm))
    ;   (cons "label_value" (position *target-object-name* label-names :test #'string=)))
    ; (send *ri* :set-object-segmentation-candidates
    ;       :arm arm
    ;       :candidates (mapcar #'(lambda (x)
    ;                               (position x label-names :test #'string=))
    ;                           (append (list "background") tote-contents)))
    ;; add kiva_pod and tote as collision object
    (add-kiva-pod-scene)
    (add-tote-scene)

    (ros::ros-info "recognizing objects in tote")
    (send *ri* :angle-vector-raw (send *baxter* :rotate-gripper arm 90 :relative nil))
    (send *ri* :angle-vector-raw
          (if (eq arm :rarm)
            (send *baxter* arm :inverse-kinematics
                          (make-coords :pos #f(600 -150 -50)
                                       :rpy (list 0 pi/2 0)))
            (send *baxter* arm :inverse-kinematics
                          (make-coords :pos #f(600 150 -50)
                                       :rpy (list 0 pi/2 0))))
          3000 arm-controller 0 :scale 3.0)

    (send *ri* :wait-interpolation)
    (send *ri* :recognize-objects-in-order-bin :arm arm :stamp (ros::time-now))
    ; delete tote as collision object
    (send *co* :delete-object *tote*)
    ; (send *baxter* :place-object-pose arm)
    ; (send *baxter* arm :move-end-pos #f(0 0 100) :world)
    ; (send *ri* :angle-vector (send *baxter* :angle-vector) 3000 arm-controller 0 :move-arm arm)
    ; (send *ri* :wait-interpolation)
    (send *ri* :pick-object-in-order-bin arm
          :n-trial 2
          :n-trial-same-pos 1
          :do-stop-grasp nil
          :object-index 0)
    (setq graspingp (send *ri* :graspingp arm))
    (ros::ros-info-green "graspingp: ~a" graspingp)

    (if graspingp
      (progn
        ;; add object as attached collision object
        (send *co* :add-attached-object *target-object* (format nil "~a_gripper_vacuum_pad_base" (arm2str arm))
              :frame-id (format nil "~a_gripper_vacuum_pad_base" (arm2str arm))
              :object-id "target_object"
              :relative-pose *target-object-coords*
              :touch-links (list (format nil "~a_gripper_vacuum_pad" (arm2str arm))
                                 (format nil "~a_gripper_vacuum_pad" (arm2str opposite-arm))
                                 (format nil "~a_gripper_vacuum_pad_base" (arm2str arm))
                                 (format nil "~a_gripper_vacuum_pad_base" (arm2str opposite-arm))
                                 (format nil "~a_gripper_base" (arm2str arm))
                                 (format nil "~a_gripper_base" (arm2str opposite-arm))))
        ;; add tote as collision object
        (add-tote-scene)
        ;; collect data before stow

        (send *ri* :angle-vector
              #f(0.0 97.4991 -46.626 -92.1973 65.896 -32.6953 110.632 174.199 69.997 -49.8132 23.5229 78.9504 119.157 -44.4244 13.9778 -74.2141 90.0)
              3000 opposite-arm-controller 0 :move-arm opposite-arm)

        ;; data collection configs
        (setq localtime (unix::localtime))
        (setq date (format nil "~A~0,2d~0,2d"
                           (+ (elt localtime 5) 1900) (+ (elt localtime 4) 1) (elt localtime 3)))
        (setq save-dir
              (format nil "~A_~A_stow/trial_~A_~0,2d~0,2d~0,2d"
                      date (if is-bimanual "dualarm" "singlearm")
                      date (elt localtime 2) (elt localtime 1) (elt localtime 0)))
        ; (ros::set-dynparam
        ;   (format nil "/before_stow_data_collection")
        ;   (cons "save_dir" (format nil "~A/~A" save-dir "before_stow")))
        (ros::set-dynparam
          (format nil "/after_stow_data_collection")
          (cons "save_dir" (format nil "~A/~A" save-dir "after_stow")))
        (ros::set-dynparam
          (format nil "/timer_stow_data_collection")
          (cons "save_dir" (format nil "~A/~A" save-dir "during_stow")))
        (if (send
              (ros::service-call (format nil "/timer_stow_data_collection/start_request")
                                 (instance std_srvs::TriggerRequest :init))
              :success)
          (ros::ros-info "[collect-data-during-stow] success to start collecting data")
          (ros::ros-error "[collect-data-during-stow] failed to start collecting data"))

        (if (eq arm :larm)
          (send *baxter* :head_pan :joint-angle -80) ;; larm
          (send *baxter* :head_pan :joint-angle 80)) ;; rarm
        (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 1000 :head-controller 0)

        ;; PARAMS
        ;; (setq entrance-x-list (list -100 -50 0 150))
        (setq entrance-x-list (list -100 -50 0 100 200))
        (setq entrance-x-list-with-opposite-arm (list -100 -50 -20))
        (setq entrance-x-list-without-opposite-arm (list 0 100 200))
        (setq offset-avoid-bin-top 30)
        (setq offset-avoid-bin-top-exit 50)
        (setq gripper-angle 45)
        (setq opposite-arm-distance 80)
        ;; PARAMS

        (setq path-constraints
              (instance moveit_msgs::Constraints :init
                        :name ""
                        :orientation_constraints
                        (list (instance moveit_msgs::OrientationConstraint :init
                                        :header
                                        (instance std_msgs::header :init :frame_id
                                                  (format nil "~a_gripper_vacuum_pad_base" (arm2str arm)))
                                        :link_name (format nil "~a_gripper_vacuum_pad_base" (arm2str arm))
                                        :orientation
                                        (instance geometry_msgs::Quaternion :init
                                                              :x 0 :y 0 :z 0 :w 1)))))

        (send *ri* :angle-vector
              (send *ri* :ik->bin-entrance arm *target-bin*
                    :offset (float-vector -150 0 offset-avoid-bin-top)
                    :gripper-angle 90)
              3000 arm-controller 0 :move-arm arm
              :path-constraints path-constraints)
        (send *ri* :wait-interpolation)
        (unix::sleep 1)
        (if (send
              (ros::service-call (format nil "/timer_stow_data_collection/end_request")
                                 (instance std_srvs::TriggerRequest :init))
              :success)
          (ros::ros-info "[collect-data-during-stow] success to end collecting data")
          (ros::ros-error "[collect-data-during-stow] failed to end collecting data"))

        ;; get result from alexnet
        (print "subscribing")
        (if (eq is-bimanual :selective)
          (progn
            (setq cls-msg
                  (one-shot-subscribe
                    "/alex_proba/output"
                    jsk_recognition_msgs::ClassificationResult
                    :after-stamp (ros::time-now)
                    :timeout 20000))
            (setq is-bimanual (evaluate-cls-msg cls-msg))
            (send *baxter* :fold-pose-back opposite-arm)))

        (ros::ros-info-green "is bimanual stow?: ~a" is-bimanual)
        (ros::set-param
          (format nil "~a_hand/is_bimanual" (arm2str arm)) is-bimanual)

        ;; bimanual hold
        (if is-bimanual
          (progn
            (send *ri* :angle-vector-raw (send *baxter* :approaching-from-downside-pose opposite-arm) 3000 opposite-arm-controller)
            (send *ri* :wait-interpolation)
            (send *ri* :angle-vector
                  (send *baxter* opposite-arm :inverse-kinematics
                        (send *baxter* :get-support-arm-coords arm
                              :dz (- (+ opposite-arm-distance 300)))
                        :use-gripper t
                        :rotation-axis :z
                        :revert-if-fail nil)
                  3000 opposite-arm-controller 0 :move-arm opposite-arm)
            (send *ri* :wait-interpolation)
            (send *ri* :start-grasp opposite-arm)
            (send *ri* :angle-vector
                  (send *baxter* opposite-arm :inverse-kinematics
                        (send *baxter* :get-support-arm-coords arm
                              :dz (- opposite-arm-distance))
                        :use-gripper t
                        :rotation-axis :z
                        :revert-if-fail nil)
                  3000 opposite-arm-controller 0 :move-arm opposite-arm)
            ;; wait for :interpolatingp
            ; (unix::usleep 1000)
            ; (while (send *ri* :interpolatingp)
            ;   (if (send *ri* :graspingp opposite-arm)
            ;     (progn
            ;       (ros::ros-info "Grasping detected. Cancel angle vector: ~a" opposite-arm)
            ;       (send *ri* :cancel-angle-vector
            ;             :controller-type opposite-arm-controller)))
            ;   (unix::usleep (* 10 1000)))
            (send *ri* :wait-interpolation)

            ; (setq opposite-arm-graspingp (send *ri* :graspingp opposite-arm))
            ; (ros::ros-info-green (format nil "opposite-arm-graspingp ~a" opposite-arm-graspingp))
            ; (unless opposite-arm-graspingp
            ;   (send *ri* :stop-grasp opposite-arm))
            (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
            (setq opposite-arm-distance
                  (- (elt (send (send *baxter* arm :end-coords) :worldpos) 2)
                     (elt (send (send *baxter* opposite-arm :end-coords) :worldpos) 2))) (ros::ros-info-green (format nil "opposite-arm-distance: ~a" opposite-arm-distance))

            ;; delete kiva_pod as collision object
            (send *co* :delete-object *kiva-pod*)

            ;; place_object
            ; (setq avs-picked->place-bin-with-opposite-arm
            ;       (mapcar
            ;         #'(lambda (x)
            ;             (send *ri* :ik->bin-entrance arm *target-bin*
            ;                   :offset (float-vector x 0 offset-avoid-bin-top)
            ;                   :gripper-angle 90)
            ;             (send *baxter* opposite-arm :inverse-kinematics
            ;                   (send *baxter* :get-support-arm-coords arm
            ;                         :dz (- opposite-arm-distance))
            ;                   :use-gripper t
            ;                   :rotation-axis :z
            ;                   :revert-if-fail nil))
            ;         entrance-x-list-with-opposite-arm))
            ; (send *ri* :angle-vector-sequence-raw
            ;       avs-picked->place-bin-with-opposite-arm :fast nil 0 :scale 5.0)
            ; (send *ri* :wait-interpolation)
            (send *ri* :ik->bin-entrance arm *target-bin*
                  :offset (float-vector -20 0 offset-avoid-bin-top)
                  :gripper-angle 90)
            (send *baxter* opposite-arm :inverse-kinematics
                  (send *baxter* :get-support-arm-coords arm
                        :dz (- opposite-arm-distance))
                  :use-gripper t
                  :rotation-axis :z
                  :revert-if-fail nil)
            (send *ri* :angle-vector (send *baxter* :angle-vector) 3000 nil 0 :move-arm :arms)
            (send *ri* :wait-interpolation)
            (send *ri* :stop-grasp opposite-arm)
            (send *ri* :angle-vector-raw
                  (send *ri* :ik->bin-entrance arm *target-bin*
                       :offset (float-vector (car (last entrance-x-list-with-opposite-arm)) 0 offset-avoid-bin-top)
                       :gripper-angle gripper-angle)
                  3000 arm-controller 0 :move-arm arm)
            (send *ri* :wait-interpolation)

            ; (setq opposite-arm-wrist-angle
            ;       (if (eq arm :rarm)
            ;         (if (> opposite-arm-distance 75) -10 -20)
            ;         (if (> opposite-arm-distance 75) 10 20)))
            ; (setq avs-picked->place-bin-without-opposite-arm
            ;       (mapcar
            ;         #'(lambda (x)
            ;             (send *ri* :ik->bin-entrance arm *target-bin*
            ;                   :offset (float-vector x 0 offset-avoid-bin-top)
            ;                   :gripper-angle gripper-angle)
            ;             (send *baxter* :rotate-wrist opposite-arm opposite-arm-wrist-angle))
            ;         entrance-x-list-without-opposite-arm))
            ; (send *ri* :angle-vector-sequence-raw
            ;       avs-picked->place-bin-without-opposite-arm :fast arm-controller 0 :scale 5.0)
            ; (send *ri* :wait-interpolation)
            ; (setq opposite-arm-wrist-angle
            ;       (if (eq arm :rarm)
            ;         (if (> opposite-arm-distance 75) -30 -60)
            ;         (if (> opposite-arm-distance 75) 30 60)))
            (send *ri* :ik->bin-entrance arm *target-bin*
                  :offset (float-vector 200 0 offset-avoid-bin-top)
                  :gripper-angle gripper-angle)
            ; (send *baxter* :rotate-wrist opposite-arm opposite-arm-wrist-angle)
            (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 nil 0 :move-arm :arms)
            (send *ri* :wait-interpolation)
            ; (when opposite-arm-graspingp
            ;   (send *ri* :stop-grasp opposite-arm))

            (setq dropped (not (send *ri* :graspingp arm)))
            (if (not dropped)
              (ros::ros-info-green "arm ~a: place object ~a" arm *target-object-name*)
              (ros::ros-warn "arm ~a: dropped object" arm))
            (ros::set-param (format nil "~a_hand/dropped" (arm2str arm)) dropped)
            (send *ri* :stop-grasp arm)
            (send *ri* :spin-off-by-wrist arm :times 20)
            ; (send *ri* :wait-interpolation)
            ; (send *ri* :spin-off-by-wrist opposite-arm :times 20)
            (send *ri* :wait-interpolation))
          (progn
            ;; delete kiva_pod as collision object
            (send *co* :delete-object *kiva-pod*)

            ;; place_object
            (setq avs-picked->place-bin
                  (mapcar
                    #'(lambda (x)
                        (send *ri* :ik->bin-entrance arm *target-bin*
                              :offset (float-vector x 0 offset-avoid-bin-top)
                              :gripper-angle gripper-angle))
                    entrance-x-list))

            ;; TODO: arm collides with kiva pod
            ;; without moveit
            (send *ri* :angle-vector-sequence-raw avs-picked->place-bin :fast arm-controller 0 :scale 5.0)
            ; ===========
            ; (send *ri* :angle-vector-sequence avs-picked->place-bin 3000 arm-controller 0 :move-arm arm)
            ;; with moveit

            (send *ri* :wait-interpolation)
            (setq dropped (not (send *ri* :graspingp arm)))
            (if (not dropped)
              (ros::ros-info-green "arm ~a: place object ~a" arm *target-object-name*)
              (ros::ros-warn "arm ~a: dropped object" arm))
            (ros::set-param (format nil "~a_hand/dropped" (arm2str arm)) dropped)
            (send *ri* :stop-grasp arm)
            (send *ri* :spin-off-by-wrist arm :times 20)
            (send *ri* :wait-interpolation)))


        ;; delete object as attached collision object
        (send *co* :delete-attached-object *target-object*)

        ;; return_from_bin
        (send *baxter* :rotate-gripper arm 0 :relative nil)
        (setq avs-place-bin->exit
              (mapcar
                #'(lambda (x)
                    (send *ri* :ik->bin-entrance arm *target-bin*
                          :offset (float-vector x 0 offset-avoid-bin-top-exit)
                          :gripper-angle 0))
                (reverse entrance-x-list)))
        (send *ri* :angle-vector-sequence-raw
              avs-place-bin->exit
              :fast arm-controller 0 :scale 5.0)
        (send *ri* :wait-interpolation)

        ;; add kiva_pod as collision object
        (add-kiva-pod-scene)

        ;; return opposite-arm to fold-pose-back
        (when is-bimanual
          ; (send *ri* :angle-vector-raw (send *baxter* :rotate-wrist opposite-arm (- opposite-arm-wrist-angle)))
          (send *ri* :wait-interpolation)
          (send *ri* :angle-vector (send *baxter* :fold-pose-back opposite-arm)
                3000 opposite-arm-controller 0 :move-arm opposite-arm)
          (send *ri* :wait-interpolation))

        (if (send
              (ros::service-call (format nil "/after_stow_data_collection/save_request")
                                 (instance std_srvs::TriggerRequest :init))
              :success)
          (ros::ros-info "[collect-data-after-stow] success to collect data")
          (ros::ros-error "[collect-data-after-stow] failed to collect data"))


        ;; TODO: cannot solve path
        ;; without moveit
        (setq avs-overlook->fold-pose-back
              (list
                (send *ri* :ik->bin-entrance arm *target-bin*
                      :offset (float-vector -200 0 offset-avoid-bin-top-exit)
                      :gripper-angle 90)))
        (pushback (send *baxter* :avoid-shelf-pose arm *target-bin*) avs-overlook->fold-pose-back)
        (pushback (send *baxter* :fold-to-keep-object arm) avs-overlook->fold-pose-back)
        (pushback (send *baxter* :fold-pose-back arm) avs-overlook->fold-pose-back)
        (send *ri* :angle-vector-sequence-raw avs-overlook->fold-pose-back :fast arm-controller 0 :scale 4.0)
        ; ===========
        ; (send *ri* :angle-vector (send *baxter* :fold-pose-back arm) 3000 arm-controller 0 :move-arm arm)
        ; with moveit

        (send *ri* :wait-interpolation))
      (progn
        (send *ri* :stop-grasp arm)
        (send *ri* :angle-vector-raw (send *baxter* :fold-pose-back arm) 3000 arm-controller 0)
        (send *ri* :wait-interpolation)))
    (send *co* :wipe-all)))

(defun singlearm-stow (arm)
  (selective-stow arm nil))

(defun dualarm-stow (arm)
  (selective-stow arm t))

(defun selective-dualarm-stow (arm)
  (selective-stow arm :selective))

(warn "~% env ~%")
(warn (format nil "target object: ~a~%" *target-object-name*))
(warn (format nil "target bin: ~a~%" *target-bin*))
(warn "~% Commands ~%")
(warn "(selective-stow-init)~%")
(warn "(singlearm-stow :rarm)~%")
(warn "(dualarm-stow :rarm)~%")
(warn "(selective-dualarm-stow :rarm)~%")
