"""target pick region data"""
(defclass target-pick-region
  :super propertied-object
  :slots (a b c d e f g h i j k l))
(defmethod target-pick-region
  (:init ()
    """Local coordinates of each region
    in the podlowres objects"""
    (setq a #f(-270 1660 500))
    (setq b #f(   0 1660 500))
    (setq c #f( 270 1660 500))
    (setq d #f(-270 1430 500))
    (setq e #f(   0 1430 500))
    (setq f #f( 270 1430 500))
    (setq g #f(-270 1200 500))
    (setq h #f(   0 1200 500))
    (setq i #f( 270 1200 500))
    (setq j #f(-270  950 500))
    (setq k #f(   0  950 500))
    (setq l #f( 270  950 500)))
  (:target-list ()
    (list a b c d e f g h i j k l))
  (:target-name-list ()
    '(a b c d e f g h i j k l)))

