;;
;; common.l
;;
(load "package://jsk_2014_picking_challenge/euslisp/utils.l")

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
(load "package://jsk_2014_picking_challenge/euslisp/order-bin.l")


(defun model-init ()
  (baxter-init)
  (send *baxter* :locate #f(0 0 950) :world)
  (send *baxter* :angle-vector (send *ri* :state :potentio-vector))
  (pod-init)
  (orderbin-init)
  (objects (list *baxter* *pod* *orderbin*)))
