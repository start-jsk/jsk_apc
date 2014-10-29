(load "package://baxtereus/baxter-interface.l")


(defun init ()
  (baxter-init)
  (objects (list *baxter*)))


(defun test ()
  (send *baxter* :larm :shoulder-y :joint-angle -30)
  (send *baxter* :larm :shoulder-p :joint-angle -30)
  (send *baxter* :larm :elbow-r  :joint-angle 30)
  (send *baxter* :larm :elbow-p  :joint-angle 90)
  (send *baxter* :larm :wrist-r  :joint-angle 30)
  (send *baxter* :larm :wrist-p  :joint-angle -30)
  (send *baxter* :larm :neck-p   :joint-angle -30)
  (objects (list *baxter*))
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  )


