;; vim: set ft=lisp:
(require "package://jsk_2016_01_baxter_apc/euslisp/jsk_2016_01_baxter_apc/baxter-interface.l")


(jsk_2016_01_baxter_apc::baxter-init)

(objects (list *baxter*))

(setq arm :larm)
(if (eq arm :larm)
  (setq bin-list (list :a :b :d :e :g :h :j :k))
  (setq bin-list (list :b :c :e :f :h :i :k :l))
  )

(send *ri* :angle-vector (send *baxter* :fold-pose-back arm) 4000)
(send *ri* :gripper-servo-on arm)
(send *ri* :wait-interpolation)

(send *ri* :recognize-bin-boxes)

(dolist (bin bin-list)
  (ros::ros-info "Viewing bin ~a with arm ~a" bin arm)
  (unix::sleep 1)
  ;;
  (send *ri* :move-arm-body->bin arm bin)
  (send *irtviewer* :redraw)
  (send *ri* :wait-interpolation)
  ;;
  (send *ri* :angle-vector (send *baxter* :avoid-shelf-pose arm bin) 4000)
  (send *irtviewer* :redraw)
  (send *ri* :wait-interpolation)
  ;;
  (send *ri* :angle-vector (send *baxter* :view-hand-pose arm bin) 4000)
  (send *irtviewer* :redraw)
  (send *ri* :wait-interpolation)
  ;;
  (unix::sleep 3)
  ;;
  (send *ri* :angle-vector (send *baxter* :fold-to-keep-object arm) 4000)
  (send *irtviewer* :redraw)
  (send *ri* :wait-interpolation)
  )
