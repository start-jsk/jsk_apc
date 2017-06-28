;; vim: set ft=lisp:

;; ------------------------------------------------------------------------------------------------
;; Dynamic reconfigure util
;; ------------------------------------------------------------------------------------------------

(ros::load-ros-manifest "dynamic_reconfigure")

(defun ros::set-dynparam (node params)
  "Set dynamic reconfigure's parameter. Usage is as below:

   (setq params (list (cons \"file_name\" \"new_image.jpg\")))
   (ros::set-dynparam \"image_publisher\" params)
  "
  (let ((config (instance dynamic_reconfigure::Config :init))
        (req (instance dynamic_reconfigure::ReconfigureRequest :init)))
    (cond
      ((not (consp params)) (setq params (list params)))
      ((<= (length params) 0) (return-from 'ros::set-dynparam t))
      ((not (listp (elt params 0))) (setq params (list params)))
      )
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


;; ------------------------------------------------------------------------------------------------
;; Logging Utils
;; ------------------------------------------------------------------------------------------------

(defun ros::ros-info-red (fmt &rest args)
  (ros::ros-info "~C[3~Cm~A~C[0m" #x1b 49 (apply #'format nil fmt args) #x1b)
  )

(defun ros::ros-info-green (fmt &rest args)
  (ros::ros-info "~C[3~Cm~A~C[0m" #x1b 50 (apply #'format nil fmt args) #x1b)
  )

(defun ros::ros-info-yellow (fmt &rest args)
  (ros::ros-info "~C[3~Cm~A~C[0m" #x1b 51 (apply #'format nil fmt args) #x1b)
  )

(defun ros::ros-info-blue (fmt &rest args)
  (ros::ros-info "~C[3~Cm~A~C[0m" #x1b 52 (apply #'format nil fmt args) #x1b)
  )

(defun ros::ros-info-purple (fmt &rest args)
  (ros::ros-info "~C[3~Cm~A~C[0m" #x1b 53 (apply #'format nil fmt args) #x1b)
  )

(defun ros::ros-info-cyan (fmt &rest args)
  (ros::ros-info "~C[3~Cm~A~C[0m" #x1b 54 (apply #'format nil fmt args) #x1b)
  )


;; ------------------------------------------------------------------------------------------------
;; Json Utils
;; ------------------------------------------------------------------------------------------------

(require :json-decode "package://roseus_mongo/euslisp/json/json-decode.l")

(defun read-file-into-string (filename)
  (with-open-file (f filename :direction :input)
    (let (buf (str-list nil))
      (while (setq buf (read-line f nil))
             (setq str-list (append str-list (list (format nil buf)))))
      (reduce #'(lambda (x y) (concatenate string x y)) str-list))))

(defun read-json-into-string (filename)
  (if (substringp "package://" filename)
    (setq filename (ros::resolve-ros-path filename)))
  (json::parse-from-string (read-file-into-string filename)))
