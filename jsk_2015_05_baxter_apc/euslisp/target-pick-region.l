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
    (setq a #f(-270 1660 550))
    (setq b #f(   0 1660 550))
    (setq c #f( 270 1660 550))
    (setq d #f(-270 1430 550))
    (setq e #f(   0 1430 550))
    (setq f #f( 270 1430 550))
    (setq g #f(-270 1200 550))
    (setq h #f(   0 1200 550))
    (setq i #f( 270 1200 550))
    (setq j #f(-270  950 550))
    (setq k #f(   0  950 550))
    (setq l #f( 270  950 550)))
  (:target-list ()
    (list a b c d e f g h i j k l))
  (:target-name-list ()
    '(a b c d e f g h i j k l)))

