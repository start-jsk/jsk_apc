(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(setq work-order-left nil)
(setq work-order-right nil)
(defun work-order-cb-left (msg)
  (dolist (data (send msg :array))
    (pushback (list "left" (str-to-symbol (send data :bin)) (send data :object)) work-order-left)))
(defun work-order-cb-right (msg)
  (dolist (data (send msg :array))
    (pushback (list "right" (str-to-symbol (send data :bin)) (send data :object)) work-order-right)))

(setq work-order nil)
(defun compose-work-order ()
  (while
    (or (> (length work-order-left) 0) (> (length work-order-right) 0))
    (when (> (length work-order-left) 0) (pushback (pop work-order-left) work-order))
    (when (> (length work-order-right) 0) (pushback (pop work-order-right) work-order)))
  work-order)

(ros::subscribe "/work_order/left_limb" jsk_2014_picking_challenge::WorkOrderArray #'work-order-cb-left)
(ros::subscribe "/work_order/right_limb" jsk_2014_picking_challenge::WorkOrderArray #'work-order-cb-right)

(defun get-work-order ()
  (ros::spin-once)
  (compose-work-order)
  work-order)