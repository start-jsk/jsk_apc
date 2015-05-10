;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

(setq *get-work-order* (make-hash-table))
(defun work-order-cb (msg arm)
  (let (work-order bin object)
    (dolist (data (send msg :array))
      (setq bin (str-to-key (send data :bin)))
      (setq object (send data :object))
      (pushback (list bin object) work-order))
    (setf (gethash arm *get-work-order*) work-order)
    ))

(defun get-work-order (arm)
  (unless (ros::get-topic-subscriber (format nil "work_order/~A_limb" (arm-to-str arm)))
    (ros::subscribe (format nil "work_order/~A_limb" (arm-to-str arm)) jsk_2014_picking_challenge::WorkOrderArray
                    #'(lambda (msg) (work-order-cb msg arm)))
    (unix::sleep 1))
  (ros::spin-once)
  (gethash arm *get-work-order*))
