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
    (setq a #f(-290 1710 570))
    (setq b #f(   0 1710 570))
    (setq c #f( 290 1710 570))
    (setq d #f(-290 1500 570))
    (setq e #f(   0 1500 570))
    (setq f #f( 290 1500 570))
    (setq g #f(-290 1230 570))
    (setq h #f(   0 1230 570))
    (setq i #f( 290 1230 570))
    (setq j #f(-290 990 570))
    (setq k #f(   0 990 570))
    (setq l #f( 290 990 570)))
  (:target-list ()
    (list a b c d e f g h i j k l))
  (:target-name-list ()
    '(a b c d e f g h i j k l)))

