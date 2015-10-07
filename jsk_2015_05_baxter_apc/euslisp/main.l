#!/usr/bin/env roseus
;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(ros::roseus "robot_main")

(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-utils.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-init.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-input.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-motions.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-visualization.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-recognition.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-communication.l")

(defun main (arm)
  (bounding-box-array-in-bin :timeout 3)
  (while  ;; wait for user input to start
    (equal (ros::get-param (format nil "~A_process/state" (arm-to-str arm))) "wait_for_user_input")
    (ros::ros-info "waiting for user input to start: ~A" arm)
    (unix::sleep 1))
  (let (target param-target state work-order work)
    (setq work-order (get-work-order arm))
    (tweet-with-image (format nil "Ok, ~A start with ~A" arm (mapcar #'cadr work-order)) (arm-to-str arm))

    ; get previous target from parameter server
    ; this does work when this robot_main node respawned
    (while (not (ros::get-param (format nil "~A_process/target" (arm-to-str arm)))) (unix::sleep 1))
    (setq param-target (str-to-key (ros::get-param (format nil "~A_process/target" (arm-to-str arm)))))
    ; remove already picked target bins
    ; this does work when this robot_main node respawned
    (while
      t
      (setq work (pop work-order))
      (when (eq (elt work 0) param-target) (return)))
    ; loop of motions
    (setq n-tried 0)  ;; how many tried to pick
    (speak-en "Let's start picking challenge" :google t)
    (while
      work
      ;; get status from parameter server
      ;; (speak-en  (format nil "Next Target is ~A" "OREO") :google t  :wait t)
      (setq target (str-to-key (elt work 0)) target-object (elt work 1))
      (ros::set-param (format nil "~A_process/target" (arm-to-str arm)) (key-to-str target))
      (setq state (ros::get-param (format nil "~A_process/state" (arm-to-str arm))))
      (ros::ros-info "state: ~A, arm: ~A, target: ~A, object: ~A"
                     state (arm-to-str arm) (key-to-str target) target-object)
      (publish-state-info state arm)
      (publish-overlay-text (format nil "~A (Left ~A objects)" (string-upcase (arm-to-str arm)) (length work-order))
			    (format nil "~A_fixed_info" (arm-to-str arm))
			    *overlay-text-width* *overlay-text-height* 0 (if (equal "left" (arm-to-str arm)) 0 (* 3 *overlay-text-height*) ):text-size 20)
      (publish-work-order-list (arm-to-str arm) work-order)
      (cond
        ((string= state "pick_object")
         ;; wait for opposite arm if opposite arm's target is center bin
         (while
           (eq
             (which-bin-region (str-to-key (ros::get-param (format nil "~A_process/target" (arm-to-str (opposite-arm arm))))))
             :center)
           (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "wait_for_opposite")
           (ros::ros-info "waiting for opposite arm to finish one order for center bin")
           (publish-state-info "Waiting for opposite arm to finish center bin" arm)
           (move-to-wait-pose arm target)
           (unix::sleep 1))
         (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "pick_object")
         ;; wait for opposite arm if current arm's target is center bin and opposite is still processing
         (while
           (and
             (eq (which-bin-region target) :center)
             (not (string= (ros::get-param (format nil "~A_process/state" (arm-to-str (opposite-arm arm)))) "wait_for_opposite")))
           (ros::ros-info "waiting for opposite arm to do order for center bin")
           (publish-state-info "Waiting for opposite arm to start center bin" arm)
           (unix::sleep 1))
         (ros::ros-info "Move to Bin ~A. Target is ~A." (key-to-str target) target-object)
         (publish-overlay-text (format nil "    PICK: ~A IN ~A" target-object (key-to-str target))
                               (format nil "~A_target_info" (arm-to-str arm))
                               *overlay-text-width* *overlay-text-height* 0 (if (equal "left" (arm-to-str arm)) (* *overlay-text-height* 2) (* *overlay-text-height* 5)) :text-size 20)
         (speak-en (format nil "Move to Bin ~A. Target is ~A." (key-to-str target) (underscore-to-space target-object)) :google t)
         (send *baxter* :locate #f(0 0 0) :world)  ;; TODO: historical reason
         (send *ri* :move-arm-body->bin arm target)
         (send *baxter* :locate #f(0 0 950) :world)  ;; TODO: historical reason
         (speak-en (format nil "Pick Object in ~A." (key-to-str target)) :google t)
         (if (and  ;; decide if going to see if grabbed
               (member target-object (get-objects-to-see-if-grabbed))
               (= (length (get-bin-contents target)) 1)
               )
           (pick-object arm target :to-see-if-grabbed t :n-tried n-tried :bin-objects (get-bin-contents target))
           (pick-object arm target :to-see-if-grabbed nil :n-tried n-tried :bin-objects (get-bin-contents target))
           )
         (incf n-tried)  ;; how many times to try to pick
         (if (= (length (get-bin-contents target)) 1)
           (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "place_object")  ;; SINGLE-ITEM-BIN
           (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "move_for_verification")))  ;; MULTI-ITEM-BIN
        ((string= state "move_for_verification")
         (speak-en  "Move for verification." :google t)
         (move-for-verification arm target)
         (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "verify_object"))
        ((string= state "verify_object")
         (let ((correct? (verify-object arm target-object)) overlay-text-msg overlay-text-color)
           (if correct?
             (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "place_object")
             (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "return_object"))
           (if correct?
             (progn
	       (setq overlay-text-msg (format nil "    Success: ~A's PICK ~A (~A)!!" (arm-to-str arm) target-object (key-to-str target)))
               (setq overlay-text-color (instance std_msgs::ColorRGBA :init :r 0.2 :g 1 :b 0.2 :a 1))
               )
             (progn
	       (setq overlay-text-msg (format nil "    FAIL: ~A's PICK. This was not ~A (~A). I Will Return It" (arm-to-str arm) target-object (key-to-str target)))
               (setq overlay-text-color (instance std_msgs::ColorRGBA :init :r 1 :g 0.1 :b 0.1 :a 1))
               )
             )
           (publish-overlay-text overlay-text-msg (format nil "~A_target_info" (arm-to-str arm))
                                 *overlay-text-width* *overlay-text-height* 0 (if (equal "left" (arm-to-str arm)) (* *overlay-text-height* 2) (* *overlay-text-height* 5)) :text-size 20 :fg-color overlay-text-color)
           ))
        ((string= state "return_object")
         (speak-en (format nil "Return object to Bin ~A." (key-to-str target)) :google t)
         (send *baxter* :locate #f(0 0 0) :world)  ;; TODO: historical reason
         (send *ri* :move-arm-body->bin arm target)
         (send *baxter* :locate #f(0 0 950) :world)  ;; TODO: historical reason
         (return-object arm target)
         ;; if have tried N times and N is larger than number of bin contents,
         ;; robot should abort the target
         (when (<= (length (get-bin-contents target)) n-tried)
           (setq work (pop work-order) n-tried 0))
         (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "pick_object"))
        ((string= state "place_object")
         (speak-en  "Target is verified. Put in Order bin." :google t)
         (tweet-with-image (format nil "I got ~a!!" target-object) (arm-to-str arm))
         (send *ri* :move-arm-body->order-bin arm)
         (send *ri* :wait-interpolation)
         (send *ri* :spin-off-by-wrist arm)
         (send *ri* :wait-interpolation)

	 ;;publish score
	 (update-score target)
	 (tweet-score *score-points* *total-score*)
	 (publish-score *score-points* *total-score*)

         (setq work (pop work-order) n-tried 0)  ;; initialize for loop
         (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "pick_object"))
        (t (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "pick_object"))
        )
      )
    (move-to-wait-pose arm target)
    (ros::set-param (format nil "~A_process/state" (arm-to-str arm)) "wait_for_opposite")
    (ros::set-param (format nil "~A_process/target" (arm-to-str arm)) "")
    ))

(warn "~% Commands ~%")
(warn "(apc-init) : Setup~%")
(warn "(main)     : Start the loop~%")
