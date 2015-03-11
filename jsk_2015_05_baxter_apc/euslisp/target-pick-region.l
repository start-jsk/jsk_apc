;;;
;;; Target pick region in the shelf.
;;;
(defclass target-pick-region
  :super propertied-object
  :slots (a b c d e f g h i j k l))
(defmethod target-pick-region
  (:init ()
    """Local coordinates of each region
    in the podlowres objects"""
    (setq a #f(-270 1660 400))
    (setq b #f(   0 1660 400))
    (setq c #f( 270 1660 400))
    (setq d #f(-270 1430 400))
    (setq e #f(   0 1430 400))
    (setq f #f( 270 1430 400))
    (setq g #f(-270 1200 400))
    (setq h #f(   0 1200 400))
    (setq i #f( 270 1200 400))
    (setq j #f(-270  950 400))
    (setq k #f(   0  950 400))
    (setq l #f( 270  950 400)))
  (:target-list ()
    (list a b c d e f g h i j k l))
  (:target-name-list ()
    '(a b c d e f g h i j k l)))

