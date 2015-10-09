;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2015_05_baxter_apc")

;; -------------------
;; Get work order data
;; -------------------
(if (not (boundp '*work-order*))
    (setq *work-order* (make-hash-table)))
(defun work-order-cb (msg arm)
  (let (work-order bin object)
    (dolist (data (send msg :array))
      (setq bin (str-to-key (send data :bin)))
      (setq object (send data :object))
      (pushback (list bin object) work-order))
    (setf (gethash arm *work-order*) work-order)
    ))

(defun work-order-cb-larm (msg) (work-order-cb msg :larm))
(defun work-order-cb-rarm (msg) (work-order-cb msg :rarm))

(ros::subscribe "work_order/left_process"
                jsk_2015_05_baxter_apc::WorkOrderArray
                #'work-order-cb-larm)
(ros::subscribe "work_order/right_process"
                jsk_2015_05_baxter_apc::WorkOrderArray
                #'work-order-cb-rarm)

(defun get-work-order (arm)
  (ros::spin-once)
  (gethash arm *work-order*))

;; ---------------------
;; Get bin contents data
;; ---------------------
(if (not (boundp '*bin-contents*))
    (setq *bin-contents* nil))
(defun bin-contents-cb (msg) (setq *bin-contents* (send msg :array)))

(ros::subscribe "/bin_contents" jsk_2015_05_baxter_apc::BinContentsArray
                #'bin-contents-cb)
(defun get-bin-contents (bin)
  (ros::spin-once)
  (dolist (bc *bin-contents*)
    (when (string= (key-to-str bin) (send bc :bin))
      (return-from get-bin-contents (send bc :objects)))))
