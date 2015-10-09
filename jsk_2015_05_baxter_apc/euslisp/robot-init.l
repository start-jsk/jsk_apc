;; vim: set ft=lisp:
;; -*- mode: lisp;-*-

(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/model/pod-lowres.l")

(setq *debug-view* nil)
(setq *simulator-p* nil)
(defun apc-init (&optional (ctype :default-controller))
  (jsk_2015_05_baxter_apc::baxter-init :ctype ctype)
  (send *baxter* :locate #f(0 0 950) :world)  ;; baxter in the lab
  (pod-init)
  (objects (list *baxter* *pod*))
  (setq *simulator-p* (not (send *ri* :joint-action-enable)))
  t)
