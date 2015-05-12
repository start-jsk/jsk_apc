;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(setq *work-order* (make-hash-table))
(defun work-order-cb (msg arm)
  (let (work-order bin object)
    (dolist (data (send msg :array))
      (setq bin (str-to-key (send data :bin)))
      (setq object (send data :object))
      (pushback (list bin object) work-order))
    (setf (gethash arm *work-order*) work-order)
    ))

(defun get-work-order (arm)
  (ros::subscribe (format nil "work_order/~A_limb" (arm-to-str arm))
                  jsk_2014_picking_challenge::WorkOrderArray
                  #'(lambda (msg) (work-order-cb msg arm)))
  (ros::spin-once)
  (gethash arm *work-order*))
