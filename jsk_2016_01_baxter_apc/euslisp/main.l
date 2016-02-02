(ros::load-ros-manifest "baxtereus")
(load "package://baxtereus/baxter-interface.l")
(load "package://euslisp/jskeus/eus/models/arrow-object.l")

(setq *oreo* (make-cube 100 150 150 :pos #f(1200 0 1435)))
(send *oreo* :set-color :blue)
(setq tote_height 400)
(setq tote_pos (vector 780 0))
(setq table (make-cube 300 300 20))
(setq posA (make-coords :pos #f(800 0 1200)))
(setq posB (make-coords :pos #f(600 300 1000) :rpy (float-vector -pi/2 0 0)))

(defun init ()
  ;; baxter model
  (when (not (boundp '*baxter*))
    (baxter-init)
    (send table :transform (send *baxter* :larm :end-coords :worldcoords))
    (send *baxter* :larm :end-coords :assoc table)
    (send (send *baxter* :rarm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
;;    (send (send *baxter* :larm :end-coords) :newcoords (make-coords :pos #f(50 0 310) :rpy (float-vector 0 -pi/2 0)))
    (send *baxter* :locate #f(0 0 950) :world))
  ;; pod-lowres model
  (when (not (boundp '*pod*))
    (load "package://jsk_2016_01_baxter_apc/euslisp/include/pod-lowres.l")
    (pod-init))

  (when (not (boundp '*tote*))
    (setq *tote* (body- (make-cube 400 500 300)
                        (make-cube 350 450 275 :pos #f(0 0 25))))
    (send *tote* :translate (float-vector (elt tote_pos 0) (elt tote_pos 1) (+ tote_height 150)))
    (send *tote* :set-color :red))

  ;;(setq *target* (instance-cube 100 100 100))
  (when (not (boundp '*target*))
    (setq *target* (arrow))
    (send *tote* :translate #f(0 0 0)))
  )


(defun movepos
  (robot arm endcoords axis)
  (send robot arm :inverse-kinematics endcoords :rotation-axis axis :debug-view t)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  )

(defun move-arm-into-bin
  (robot arm targetcoords)
  (let ((viacoords))
    (setq viacoords (make-coords :pos (float-vector 800 (elt (send targetcoords :pos) 1) (elt (send targetcoords :pos) 2))))
    (send viacoords :rotate pi :x)
    (movepos robot arm viacoords :x))
  (movepos robot arm targetcoords :x))


;(defun tote2pod (arm)
;  (send *target* :rotate (/ pi 2.0) :y)
;  (movepos *baxter* arm (send *target* :coords) nil)
;  (send *baxter* arm :end-coords :assoc *target*)
;  (send *baxter* arm :angle-vector #f(-20.0 -25.0 40.0 60.0 20.0 80.0 0.0))
;  (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
;  (send *ri* :wait-interpolation)
;  (move-arm-into-bin *baxter* arm (make-coords :pos (float-vector 1300 (random 200) (+ 1000 (random 600)))))
;  (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
;  (send *ri* :wait-interpolation)
;  (send *baxter* arm :end-coords :dissoc *target*)
;  (send *baxter* arm :angle-vector #f(-20.0 -25.0 40.0 60.0 20.0 80.0 0.0))
;  (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
;  (send *ri* :wait-interpolation))
;
(defun movetable (goalcoords)
  (let ((tempcoords (send goalcoords :copy-coords)))
    (send tempcoords :rotate pi :x)
    (movepos *baxter* :larm tempcoords t)
    )
  )


(defun pod2reset (arm targetcoords)
  (move-arm-into-bin *baxter* arm targetcoords)
  ;;insert grasp motion
;  (send *baxter* arm :end-coords :assoc targetcoords)
  (let ((tempcoords))
    (setq tempcoords (make-coords :pos (float-vector 800 (elt (send targetcoords :pos) 1) (elt (send targetcoords :pos) 2))))
    (movepos *baxter* arm tempcoords :x))
  (send *baxter* arm :angle-vector #f(-20.0 -25.0 40.0 60.0 20.0 80.0 0.0))
  (send *ri* :angle-vector (send *baxter* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  )

(defun pod2table (arm targetcoords)
  (movetable posB)
  (pod2reset arm targetcoords)
  (movetable posA)
  (movepos *baxter* :rarm (send *baxter* :larm :end-coords) nil))

(defun test ()
  (pod2table :rarm *oreo*))

(ros::ros-info "(test) : start demo")
(init)
(movetable posB)
(objects (list *baxter* *pod* *tote* *target* *oreo* table))
