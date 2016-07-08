;; vim: set ft=lisp:

(ros::load-ros-manifest "dynamic_reconfigure")

(defun ros::set-dynparam (node params)
  "Set dynamic reconfigure's parameter. Usage is as below:

   (setq params (list (cons \"file_name\" \"new_image.jpg\")))
   (ros::set-dynparam \"image_publisher\" params)
  "
  (let ((config (instance dynamic_reconfigure::Config :init))
        (req (instance dynamic_reconfigure::ReconfigureRequest :init)))
    (dolist (param params)
      (let ((name (car param)) (value (cdr param)) (dynparam))
        (cond
          ((or (eq value t) (eq value nil))
           (setq dynparam (instance dynamic_reconfigure::BoolParameter :init))
           (send dynparam :name name)
           (send dynparam :value value)
           (send config :bools (append (send config :bools) (list dynparam))))
          ((integerp value)
           (setq dynparam (instance dynamic_reconfigure::IntParameter :init))
           (send dynparam :name name)
           (send dynparam :value value)
           (send config :ints (append (send config :ints) (list dynparam))))
          ((stringp value)
           (setq dynparam (instance dynamic_reconfigure::StrParameter :init))
           (send dynparam :name name)
           (send dynparam :value value)
           (send config :strs (append (send config :strs) (list dynparam))))
          ((floatp value)
           (setq dynparam (instance dynamic_reconfigure::DoubleParameter :init))
           (send dynparam :name name)
           (send dynparam :value value)
           (send config :doubles (append (send config :doubles) (list dynparam))))
          (t (ros::ros-error "Unexpected type of parameter. name: ~A, value: ~A" name value))
          )
        )
      )
    (send req :config config)
    (ros::service-call (format nil "~A/set_parameters" node) req)
    ) ;; let
  ) ;; defun
