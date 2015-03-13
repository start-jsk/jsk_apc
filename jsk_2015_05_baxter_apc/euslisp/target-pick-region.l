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
    (setq a #f(-270 1700 570))
    (setq b #f(   0 1700 570))
    (setq c #f( 270 1700 570))
    (setq d #f(-270 1470 570))
    (setq e #f(   0 1470 570))
    (setq f #f( 270 1470 570))
    (setq g #f(-270 1260 570))
    (setq h #f(   0 1260 570))
    (setq i #f( 270 1260 570))
    (setq j #f(-270 1020 570))
    (setq k #f(   0 1020 570))
    (setq l #f( 270 1020 570)))
  (:target-list ()
    (list a b c d e f g h i j k l))
  (:target-name-list ()
    '(a b c d e f g h i j k l)))

