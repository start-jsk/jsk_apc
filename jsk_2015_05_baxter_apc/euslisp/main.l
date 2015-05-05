#!/usr/bin/env roseus
;;

(ros::roseus "robot_main")

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-init.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-motions.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-test.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-visualization.l")

(setq *bin-contents* nil)
(ros::subscribe "/bin_contents" jsk_2014_picking_challenge::BinContentsArray
                #'(lambda (msg) (setq *bin-contents* (send msg :array))))

(defun get-bin-contents (bin)
  (ros::spin-once)
  (dolist (bc *bin-contents*)
    (when (string= (symbol-to-str bin) (send bc :bin))
      (return-from get-bin-contents (send bc :objects)))))

(defun main ()
  (let (arm target param-target state work)
    ; get previous target from parameter server
    ; this does work when this robot_main node respawned
    (while (not (ros::get-param "/target")) (unix::sleep 1))
    (setq param-target (str-to-symbol (ros::get-param "/target")))
    ; remove already picked target bins
    ; this does work when this robot_main node respawned
    (while
      t
      (setq work (pop work-order))
      (when (eq (elt work 1) param-target) (return)))
    ; loop of motions
    (speak-en "Let's start picking challenge" :google t)
    (while
      work
      ; get status from parameter server
      ;; (speak-en  (format nil "Next Target is ~A" "OREO") :google t  :wait t)
      (setq arm (str-to-arm (elt work 0)) target (str-to-symbol (elt work 1)))
      (ros::set-param "/target" (symbol-to-str target))
      (setq state (ros::get-param (format nil "~A_limb/state" (arm-to-str arm))))
      (ros::ros-info "state: ~A, arm: ~A, target: ~A" state (arm-to-str arm) (symbol-to-str target))
      (cond
        ((string= state "pick_object")
         (speak-en (format nil "Move to Bin ~A . Target is ~A" (symbol-to-str target) "OREO" ) :google t)
         (move-to-target-bin arm target)
         (speak-en (format nil "Pick Object in ~A." (symbol-to-str target)) :google t)
         (pick-object arm target)
         (if (= (length (get-bin-contents target)) 1)
          (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "place_object")
          (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "move_for_verification")))
        ((string= state "move_for_verification")
         (speak-en  "Move for verification." :google t)
         (move-for-verification arm)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "verify_object"))
        ((string= state "verify_object")
         (let ((correct? (verify-object arm (elt work 1))))
           (if correct?
             (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "place_object")
             (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "return_object"))))
        ((string= state "return_object")
         (speak-en (format nil "Return object to Bin ~A." (symbol-to-str target)) :google t)
         (move-to-target-bin arm target)
         (return-object arm)
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "pick_object"))
        ((string= state "place_object")
         (speak-en  "Target is verified. Put in Order bin." :google t)
         (place-object arm)
         (setq work (pop work-order))  ; change arm and target
         (ros::set-param (format nil "~A_limb/state" (arm-to-str arm)) "pick_object"))
        (t (ros::set-param (format nil "/~A_limb/state" (arm-to-str arm)) "pick_object"))
        )
      )
    ))

(setq work-order (get-work-order))

(warn "~% Commands ~%")
(warn "(apc-init) : Setup~%")
(warn "(main)     : Start the loop~%")
