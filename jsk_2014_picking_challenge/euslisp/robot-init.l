(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/model/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/model/order-bin.l")

(setq *simulator-p* nil)
(defun apc-init ()
  (baxter-init)
  (send *baxter* :locate #f(0 0 950) :world)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (send (send *baxter* :larm :end-coords) :newcoords (make-coords :pos #f(0 0 100) :rpy (float-vector 0 -pi/2 0)))
  (send (send *baxter* :rarm :end-coords) :newcoords (make-coords :pos #f(0 0 100) :rpy (float-vector 0 -pi/2 0)))
  (pod-init)
  (orderbin-init)
  (objects (list *baxter* *pod* *orderbin*))
  (setq *simulator-p* (not (send *ri* :joint-action-enable)))
  t)

(defun publish-end-coords ()
  (ros::advertise "/larm_end_coords" visualization_msgs::Marker 5)
  (ros::advertise "/rarm_end_coords" visualization_msgs::Marker 5)
  (ros::publish "/larm_end_coords" (eusobj->marker-msg (send *baxter* :larm :end-coords) (instance std_msgs::header :init :stamp (ros::time-now) :frame_id "/base")))
  (ros::publish "/rarm_end_coords"(eusobj->marker-msg (send *baxter* :larm :end-coords) (instance std_msgs::header :init :stamp (ros::time-now) :frame_id "/base"))))

  
