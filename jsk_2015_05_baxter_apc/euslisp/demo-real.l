#!/usr/bin/env roseus
;; demo.l
(load "package://baxtereus/baxter-interface.l")

(defun init ()
  (baxter-init)
  (send *baxter* :locate #f(0 0 950) :world)
  (objects (list *baxter*)))

(ros::roseus "demo_real")

(init)
(send *baxter* :reset-pose)
(send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)

(send *ri* :stop-grasp)

;; qr code pos
(send *ri* :angle-vector #f(1.09863 2.72461 -8.54736 -35.354 17.9297 -21.3574 96.5479 -9.51416 56.2286 2.31837 -4.20656 59.6595 16.3488 -44.1575 -2.89331) 4000)
; (send *ri* :angle-vector #f(-0.351562 2.71419 -8.56179 -35.2521 17.9237 -21.3521 96.5801 -9.57027 59.2092 -14.6541 -4.56828 93.14 14.7365 -59.9863 2.27213) 4000)
(unix:sleep 10)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)

(send *baxter* :rarm :move-end-pos #f(0 0 60) :world)
(send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)

(send *baxter* :rarm :move-end-pos #f(100 0 0) :world)
(send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)

(send *ri* :start-grasp)

(send *baxter* :rarm :move-end-pos #f(0 0 80) :world)
(send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)


(send *baxter* :rarm :move-end-pos #f(-200 0 0) :world)
(send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)

(send *baxter* :untuck-pose)
(send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)


(send *ri* :angle-vector
     #f(1.14258 19.9951 -25.0049 -39.9683 59.9414 -19.9951 79.9585 0.0 -38.2104 -32.1021 21.6431 64.0283 -21.8848 60.7324 6.48193))
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)

(send *baxter* :rarm :move-end-pos #f(0 0 -200) :world)
(send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)

(send *ri* :stop-grasp)

(send *baxter* :rarm :move-end-pos #f(0 0 200) :world)
(send *ri* :angle-vector (send *baxter* :angle-vector) 4000)
(unix:sleep 5)
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)

(send *baxter* :reset-pose)
(send *ri* :angle-vector (send *baxter* :angle-vector))
(send *baxter* :angle-vector (send *ri* :state :potentio-vector))
(send *irtviewer* :draw-objects)