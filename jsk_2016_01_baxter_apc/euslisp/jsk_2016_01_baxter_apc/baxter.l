;; vim: set ft=lisp:

(require "package://baxtereus/baxter-util.l")

(unless (find-package "JSK_2015_05_BAXTER_APC")
  (make-package "JSK_2015_05_BAXTER_APC"))

(defclass jsk_2016_01_baxter_apc::baxter-robot
  :super baxter-robot
  :slots ())
(defmethod jsk_2016_01_baxter_apc::baxter-robot
  (:init
    (&rest args)
    (send-super* :init args)
    (send (send self :rarm :end-coords)
          :newcoords (make-coords :pos #f(50 0 310)
                                  :rpy (float-vector 0 -pi/2 0)))
    (send (send self :larm :end-coords)
          :newcoords (make-coords :pos #f(50 0 310)
                                  :rpy (float-vector 0 -pi/2 0)))

    ;; hard-coded pose
    (sethash :fold-pose-upper (send self :get-val 'hard-coded-pose-table) #f(0.0 -7.80029 -1.12061 -82.9248 141.438 116.477 -6.48193 8.10791 7.80029 -1.12061 82.9248 141.438 -116.477 -6.48193 -8.10791))
    (sethash :fold-pose-middle (send self :get-val 'hard-coded-pose-table) #f(0.0 48.8672 -10.437 -80.6616 148.645 80.9033 1.38428 18.8745 -48.8672 -10.437 80.6616 148.645 -80.9033 1.38428 -18.8745))
    (sethash :fold-pose-lower (send self :get-val 'hard-coded-pose-table) #f(0.0 -7.27295 52.6465 -59.9414 91.582 123.574 13.3374 159.675 7.27295 52.6465 59.9414 91.582 -123.574 13.3374 -159.675))
    (sethash :fold-pose-back (send self :get-val 'hard-coded-pose-table) #f(0.0 97.4707 -2.39502 -94.5483 134.67 91.4062 8.70117 0.0 -97.4707 -2.39502 94.5483 134.67 -91.4062 8.70117 0.0))
    (sethash :fold-to-keep-object (send self :get-val 'hard-coded-pose-table) #f(0.0 77.1899 4.37256 -86.1548 123.904 11.8213 59.8096 74.5532 -77.1899 4.37256 86.1548 123.904 -11.8213 59.8096 -74.5532))
    (sethash :place-object-pose (send self :get-val 'hard-coded-pose-table) #f(0 5.07568 -13.6011 -44.7803 101.184 -0.725098 6.63574 88.6597 -5.07568 -13.6011 44.7803 101.184 0.725098 6.63574 -88.6597))
    (sethash :avoid-shelf-pose-a (send self :get-val 'hard-coded-pose-table)
             #f(0
                3.51562 -1.56006 -174.946 123.86 113.533 90.6372 -38.4961
                -3.51562 -1.56006 174.946 123.86 -113.533 90.6372 38.4961))
    (sethash :avoid-shelf-pose-b (send self :get-val 'hard-coded-pose-table)
             #f(0
                37.4634 -29.3774 -142.449 117.883 -82.0679 -18.8306 123.882
                -37.4634 -29.3774 142.449 117.883 82.0679 -18.8306 -123.882))
    (sethash :avoid-shelf-pose-c (send self :get-val 'hard-coded-pose-table)
             #f(0
                3.51562 -1.56006 -174.946 123.86 113.533 90.6372 -38.4961
                -3.51562 -1.56006 174.946 123.86 -113.533 90.6372 38.4961))
    (sethash :avoid-shelf-pose-d (send self :get-val 'hard-coded-pose-table)
             #f(0
                79.8926 -14.2822 -125.793 148.271 -46.3623 -4.24072 100.063
                -79.8926 -14.2822 125.793 148.271 46.3623 -4.24072 -100.063))
    (sethash :avoid-shelf-pose-e (send self :get-val 'hard-coded-pose-table)
             #f(0
                54.1406 -52.8662 -130.583 142.8 3.27393 -32.5854 27.5098
                -54.1406 -52.8662 130.583 142.8 -3.27393 -32.5854 -27.5098))
    (sethash :avoid-shelf-pose-f (send self :get-val 'hard-coded-pose-table)
             #f(0
                79.8926 -14.2822 -125.793 148.271 -46.3623 -4.24072 100.063
                -79.8926 -14.2822 125.793 148.271 46.3623 -4.24072 -100.063))
    (sethash :avoid-shelf-pose-g (send self :get-val 'hard-coded-pose-table)
             #f(0
                93.5156 10.5908 -83.3203 136.318 11.1621 44.6045 95.4272
                -93.5156 10.5908 83.3203 136.318 -11.1621 44.6045 -95.4272))
    (sethash :avoid-shelf-pose-h (send self :get-val 'hard-coded-pose-table)
             #f(0
                13.7988 32.8711 -78.772 147.48 2.87842 -45.0 122.937
                -13.7988 32.8711 78.772 147.48 -2.87842 -45.0 -122.937))
    (sethash :avoid-shelf-pose-i (send self :get-val 'hard-coded-pose-table)
             #f(0
                93.5156 10.5908 -83.3203 136.318 11.1621 44.6045 95.4272
                -93.5156 10.5908 83.3203 136.318 -11.1621 44.6045 -95.4272))
    (sethash :avoid-shelf-pose-j (send self :get-val 'hard-coded-pose-table)
             #f(0
                89.4946 6.02051 -56.0083 139.351 34.1895 22.6538 98.2397
                -89.4946 6.02051 56.0083 139.351 -34.1895 22.6538 -98.2397))
    (sethash :avoid-shelf-pose-k
             (send self :get-val 'hard-coded-pose-table)
             #f(0
                48.2959 18.6108 -51.7896 135.044 48.186 6.87744 89.3408
                -48.2959 18.6108 51.7896 135.044 -48.186 6.87744 -89.3408))
    (sethash :avoid-shelf-pose-l (send self :get-val 'hard-coded-pose-table)
             #f(0
                89.4946 6.02051 -56.0083 139.351 34.1895 22.6538 98.2397
                -89.4946 6.02051 56.0083 139.351 -34.1895 22.6538 -98.2397))

    ;; view-hand-pose
    (sethash :view-hand-pose-a (send self :get-val 'hard-coded-pose-table)
             #f(0
                1.05469 -3.33984 -174.814 42.9565 73.125 97.6245 -38.6719
                5.18555 -11.7554 163.608 80.2441 -93.0542 78.3545 60.1831))
    (sethash :view-hand-pose-b (send self :get-val 'hard-coded-pose-table)
            #f(0
                -5.18555 -11.7554 -163.608 80.2441 93.0542 78.3545 -60.1831
                5.18555 -11.7554 163.608 80.2441 -93.0542 78.3545 60.1831))
    (sethash :view-hand-pose-c (send self :get-val 'hard-coded-pose-table)
             #f(0
                -5.18555 -11.7554 -163.608 80.2441 93.0542 78.3545 -60.1831
                -1.05469 -3.33984 174.814 42.9565 -73.125 97.6245 38.6719))
    (sethash :view-hand-pose-d (send self :get-val 'hard-coded-pose-table)
             #f(0
                44.8462 -2.8125 -113.313 85.7593 10.1733 51.0205 -4.61426
                -17.5122 23.4668 143.701 125.728 84.0674 -62.7319 -109.358))
    (sethash :view-hand-pose-e (send self :get-val 'hard-coded-pose-table)
             #f(0
                17.5122 23.4668 -143.701 125.728 -84.0674 -62.7319 109.358
                -17.5122 23.4668 143.701 125.728 84.0674 -62.7319 -109.358))
    (sethash :view-hand-pose-f (send self :get-val 'hard-coded-pose-table)
             #f(0
                17.5122 23.4668 -143.701 125.728 -84.0674 -62.7319 109.358
                -44.8462 -2.8125 113.313 85.7593 -10.1733 51.0205 4.61426))
    (sethash :view-hand-pose-g (send self :get-val 'hard-coded-pose-table)
             #f(0
                41.3525 15.4248 -101.536 80.8154 20.083 65.2368 5.69092
                -30.4761 9.95361 93.999 100.503 -10.2832 33.3325 2.10937))
    (sethash :view-hand-pose-h (send self :get-val 'hard-coded-pose-table)
             #f(0
                30.4761 9.95361 -93.999 100.503 10.2832 33.3325 -2.10937
                -30.4761 9.95361 93.999 100.503 -10.2832 33.3325 2.10937))
    (sethash :view-hand-pose-i (send self :get-val 'hard-coded-pose-table)
             #f(0
                30.4761 9.95361 -93.999 100.503 10.2832 33.3325 -2.10937
                -41.3525 15.4248 101.536 80.8154 -20.083 65.2368 -5.69092))
    (sethash :view-hand-pose-j (send self :get-val 'hard-coded-pose-table)
             #f(0
                65.2368 19.5557 -78.2886 121.069 44.4946 18.3911 74.0698
               -46.3843 29.0918 71.4551 121.794 -48.4058 13.8647 -79.0356))
    (sethash :view-hand-pose-k (send self :get-val 'hard-coded-pose-table)
            #f(0
               46.3843 29.0918 -71.4551 121.794 48.4058 13.8647 79.0356
               -46.3843 29.0918 71.4551 121.794 -48.4058 13.8647 -79.0356))
    (sethash :view-hand-pose-l (send self :get-val 'hard-coded-pose-table)
            #f(0
               46.3843 29.0918 -71.4551 121.794 48.4058 13.8647 79.0356
              -65.2368 19.5557 78.2886 121.069 -44.4946 18.3911 -74.0698))
    )
  (:fold-pose-upper (&optional (arm :arms)) (send self :hard-coded-pose :fold-pose-upper arm))
  (:fold-pose-middle (&optional (arm :arms)) (send self :hard-coded-pose :fold-pose-middle arm))
  (:fold-pose-lower (&optional (arm :arms)) (send self :hard-coded-pose :fold-pose-lower arm))
  (:fold-pose-back (&optional (arm :arms)) (send self :hard-coded-pose :fold-pose-back arm))
  (:fold-to-keep-object (&optional (arm :arms)) (send self :hard-coded-pose :fold-to-keep-object arm))
  (:place-object-pose (&optional (arm :arms)) (send self :hard-coded-pose :place-object-pose arm))
  (:rotate-wrist
    (arm angle)
    (let (wrist-index av-tmp)
      (setq wrist-index (if (eq arm :larm) 7 14))
      (setq av-tmp (send *baxter* :angle-vector))
      (setf (elt av-tmp wrist-index) (+ (elt av-tmp wrist-index) angle))
      (send *baxter* :angle-vector av-tmp)))
  (:avoid-shelf-pose
    (arm bin)
    (case bin
      (:a (send self :hard-coded-pose :avoid-shelf-pose-a arm))
      (:b (send self :hard-coded-pose :avoid-shelf-pose-b arm))
      (:c (send self :hard-coded-pose :avoid-shelf-pose-c arm))
      (:d (send self :hard-coded-pose :avoid-shelf-pose-d arm))
      (:e (send self :hard-coded-pose :avoid-shelf-pose-e arm))
      (:f (send self :hard-coded-pose :avoid-shelf-pose-f arm))
      (:g (send self :hard-coded-pose :avoid-shelf-pose-g arm))
      (:h (send self :hard-coded-pose :avoid-shelf-pose-h arm))
      (:i (send self :hard-coded-pose :avoid-shelf-pose-i arm))
      (:j (send self :hard-coded-pose :avoid-shelf-pose-j arm))
      (:k (send self :hard-coded-pose :avoid-shelf-pose-k arm))
      (:l (send self :hard-coded-pose :avoid-shelf-pose-l arm))))
  (:view-hand-pose
    (arm bin)
    (case bin
      (:a (send self :hard-coded-pose :view-hand-pose-a arm))
      (:b (send self :hard-coded-pose :view-hand-pose-b arm))
      (:c (send self :hard-coded-pose :view-hand-pose-c arm))
      (:d (send self :hard-coded-pose :view-hand-pose-d arm))
      (:e (send self :hard-coded-pose :view-hand-pose-e arm))
      (:f (send self :hard-coded-pose :view-hand-pose-f arm))
      (:g (send self :hard-coded-pose :view-hand-pose-g arm))
      (:h (send self :hard-coded-pose :view-hand-pose-h arm))
      (:i (send self :hard-coded-pose :view-hand-pose-i arm))
      (:j (send self :hard-coded-pose :view-hand-pose-j arm))
      (:k (send self :hard-coded-pose :view-hand-pose-k arm))
      (:l (send self :hard-coded-pose :view-hand-pose-l arm))))
  )
