;; -*- mode: lisp;-*-

(load "package://jsk_2014_baxter_apc/euslisp/utils.l")
(load "package://jsk_2014_baxter_apc/euslisp/robot-utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_baxter_apc")

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
                jsk_2014_apc_common::WorkOrderArray
                #'work-order-cb-larm)
(ros::subscribe "work_order/right_process"
                jsk_2014_apc_common::WorkOrderArray
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

(ros::subscribe "/bin_contents" jsk_2014_apc_common::BinContentsArray
                #'bin-contents-cb)
(defun get-bin-contents (bin)
  (ros::spin-once)
  (dolist (bc *bin-contents*)
    (when (string= (key-to-str bin) (send bc :bin))
      (return-from get-bin-contents (send bc :objects)))))

;;----------------------
;; Calculate Score
;;----------------------
(setq *total-score* 0)
(setq *score-points* 0)
(defun update-score(bin)
  (ros::ros-info "update-score ~A" bin)
  (setq *score-points*
	(case (length (get-bin-contents bin))
	  (1 10) (2 15) (t 20)))
  (setq *total-score* (ros::get-param "/total_score"))
  (setq *total-score* (+ *total-score* *score-points*))
  (ros::set-param "/total_score" *total-score*)
  )


;; ---------------
;; Get object info
;; ---------------
(defun get-objects-to-see-if-grabbed ()
  (list
    "champion_copper_plus_spark_plug"
    ))
