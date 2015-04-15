;;
;;
;;
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")

(defun arm-str2symbol (arm)
  (cond ((string= "left") (setq res :larm))
        ((string= "right") 

(defun validate-arm (arm)
  (let (res)
    (cond ((string= arm "left") (setq res :larm))
          ((string= arm "right") (setq res :rarm))
          (t (progn (ros::ros-error "invalid arm valud")
                    (setq res nil))))
    res))

(defun validate-bin-name (bin-name)
  (let* ((region (instance target-pick-region :init)))
    (when (find bin-name (send region :target-list))
      (return-from validate-bin-name bin-name))
    (ros::ros-error "invalid bin name")
    nil))

; (defun move-arm-to-target-bin (arm bin-name)
;   (validation-res (and (validate-arm arm) (validate-bin-name bin-name)))
;   (unless (validate-arm arm) (return-from move-arm-to-target-bin))