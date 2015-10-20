;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")

(defparameter *larm-height-offset* 12)
(defparameter *larm-depth-angle* 5)
(defparameter *rarm-height-offset* 20)
(defparameter *rarm-depth-angle* 5)

(defun move-end-pos-with-interpolation (arm &key (x 0) (y 0) (z 0) (time 1500) (revert-if-fail nil) (rotation-axis :z) (force nil))
;  (case arm
;    (:larm (setq z (+ z *larm-height-offset* (* (sin (deg2rad *larm-depth-angle*)) x))))
;    (:rarm (setq z (+ z *rarm-height-offset* (* (sin (deg2rad *rarm-depth-angle*)) x)))))
  ;; if fail with revert-if-fail t, the robot posture is back to original pose
  (unless (send *baxter* arm :move-end-pos (float-vector x y z) :world :revert-if-fail revert-if-fail :rotation-axis rotation-axis)
    (ros::ros-error "[move-end-pos] ik error on ~A ~A ~A ~A" arm x y z)
    (return-from move-end-pos-with-interpolation nil))
  (if force
      (force-to-reach-goal :arm arm :stop 20 :threshold 2)
    (progn
      (send *irtviewer* :draw-objects)
      (send *ri* :angle-vector (send *baxter* :angle-vector) time)
      (send *ri* :wait-interpolation)
      )
    )
  t)

(defun move-end-rot-with-interpolation (arm &key (x 0) (y 0) (z 0) (time 1000) (revert-if-fail nil))
  (unless (send *baxter* arm :move-end-rot x :x :local :revert-if-fail revert-if-fail)
    (ros::ros-error "[move-end-rot] ik error on ~A x: ~A" arm x)
    (return-from move-end-rot-with-interpolation nil))
  (unless (send *baxter* arm :move-end-rot y :y :local :revert-if-fail revert-if-fail)
    (ros::ros-error "[move-end-rot] ik error on ~A y: ~A" arm y)
    (return-from move-end-rot-with-interpolation nil))
  (unless (send *baxter* arm :move-end-rot z :z :local :revert-if-fail revert-if-fail)
    (ros::ros-error "[move-end-rot] ik error on ~A z: ~A" arm z)
    (return-from move-end-rot-with-interpolation nil))
  (send *baxter* arm :move-end-rot x :x :local :revert-if-fail revert-if-fail)
  (send *baxter* arm :move-end-rot y :y :local :revert-if-fail revert-if-fail)
  (send *baxter* arm :move-end-rot z :z :local :revert-if-fail revert-if-fail)
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector (send *baxter* :angle-vector) time)
  (send *ri* :wait-interpolation)
  t)

(defmethod baxter-robot
  (:fold-pose
    ()
    (let (av)
      (send *baxter* :fold-pose-back :larm)
      (send *baxter* :fold-pose-back :rarm)
      (setq av (send *baxter* :angle-vector))
      (setf (aref av 0) 0)
      (send *baxter* :angle-vector av))))

(defmethod pod-lowres-object
  ; local coordinates of each target bin
  ; these are the entrances
  (:a () (send self :transform-vector #f(-280 1660 400)))
  (:b () (send self :transform-vector #f(   0 1660 400)))
  (:c () (send self :transform-vector #f( 280 1660 400)))
  (:d () (send self :transform-vector #f(-280 1430 400)))
  (:e () (send self :transform-vector #f(   0 1430 400)))
  (:f () (send self :transform-vector #f( 280 1430 400)))
  (:g () (send self :transform-vector #f(-280 1200 400)))
  (:h () (send self :transform-vector #f(   0 1200 400)))
  (:i () (send self :transform-vector #f( 280 1200 400)))
  (:j () (send self :transform-vector #f(-280  940 400)))
  (:k () (send self :transform-vector #f(   0  940 400)))
  (:l () (send self :transform-vector #f( 280  940 400)))
  (:bin-list () '(:a :b :c :d :e :f :g :h :i :j :k :l)))

(defun avoid-shelf-pose-for-bin (arm bin)
  (send *baxter* arm :angle-vector
        (case bin
          (:a #f(3.51562 -1.56006 -174.946 123.86 113.533 90.6372 -38.4961))
          (:b (case arm
                (:larm #f(37.4634 -29.3774 -142.449 117.883 -82.0679 -18.8306 123.882))
                (:rarm #f(-37.4634 -29.3774 142.449 117.883 82.0679 -18.8306 -123.882))))
          (:c #f(-3.51562 -1.56006 174.946 123.86 -113.533 90.6372 38.4961))
          (:d #f(79.8926 -14.2822 -125.793 148.271 -46.3623 -4.24072 100.063))
          (:e (case arm
                (:larm #f(54.1406 -52.8662 -130.583 142.8 3.27393 -32.5854 27.5098))
                (:rarm #f(-54.1406 -52.8662 130.583 142.8 -3.27393 -32.5854 -27.5098))))
          (:f #f(-79.8926 -14.2822 125.793 148.271 46.3623 -4.24072 -100.063))
          (:g #f(93.5156 10.5908 -83.3203 136.318 11.1621 44.6045 95.4272))
          (:h (case arm
                (:larm #f(13.7988 32.8711 -78.772 147.48 2.87842 -45.0 122.937))
                (:rarm #f(-13.7988 32.8711 78.772 147.48 -2.87842 -45.0 -122.937))))
          (:i #f(-93.5156 10.5908 83.3203 136.318 -11.1621 44.6045 -95.4272))
          (:j #f(89.4946 6.02051 -56.0083 139.351 34.1895 22.6538 98.2397))
          (:k (case arm
                (:larm #f(48.2959 18.6108 -51.7896 135.044 48.186 6.87744 89.3408))
                (:rarm #f(-48.2959 18.6108 51.7896 135.044 -48.186 6.87744 -89.3408))))
          (:l #f(-89.4946 6.02051 56.0083 139.351 -34.1895 22.6538 -98.2397))
          ))
  (send *baxter* :angle-vector))

(defun move-to-wait-pose (arm bin)
  (let ((avs)
        (diff1 (norm (v- (send *ri* :state :potentio-vector) (avoid-shelf-pose-for-bin arm bin))))
        (diff2 (norm (v- (send *ri* :state :potentio-vector) (send *baxter* :fold-pose)))))
    (if (> diff1 diff2)
      (pushback (send *baxter* :fold-pose) avs)
      (progn (pushback (avoid-shelf-pose-for-bin arm bin) avs)
             (pushback (send *baxter* :fold-pose) avs)))
    (send *ri* :angle-vector-sequence avs)
    (send *ri* :wait-interpolation)
    avs))
