;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/utils.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/robot-input.l")

(ros::load-ros-manifest "std_msgs")

(defun tweet (str)
  (ros::advertise-if-yet "/tweet_string" std_msgs::string 1)
  (ros::ros-info "tweet ~A" str)
  (ros::publish "/tweet_string" (instance std_msgs::string :init :data str)))

(defun tweet-with-image (str l/r) ;; l/r is "left" or "right"
  (let (tname)
    (cond ((equal l/r "left")
           (setq tname "/tweet_larm"))
          ((equal l/r "right")
           (setq tname "/tweet_rarm"))
          (t
           (ros::ros-error "arm name ~A is not found" l/r)
           (return-from tweet-with-image nil)))
    (ros::advertise-if-yet tname std_msgs::string 1)
    (ros::ros-info "tweet ~A to ~A" str tname)
    (ros::publish tname (instance std_msgs::string :init :data str))))

(defun tweet-score (score-points total-score)
  (tweet (format nil "Yay!! Baxter got ~A points now! Total is ~A in the challenge!!"
		 score-points total-score)))
