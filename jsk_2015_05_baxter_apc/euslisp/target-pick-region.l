;;
;; Target pick region in the shelf.
;;
(defclass target-pick-region
  :super propertied-object
  :slots ())
(defmethod target-pick-region
  ; Local coordinates of each region
  ; in the podlowres objects
  (:init () nil)
  (:a () #f(-290 1710 570))
  (:b () #f(   0 1710 570))
  (:c () #f( 290 1710 570))
  (:d () #f(-290 1500 570))
  (:e () #f(   0 1500 570))
  (:f () #f( 290 1500 570))
  (:g () #f(-290 1230 570))
  (:h () #f(   0 1230 570))
  (:i () #f( 290 1230 570))
  (:j () #f(-290 990 570))
  (:k () #f(   0 990 570))
  (:l () #f( 290 990 570))
  (:list () '(:a :b :c :d :e :f :g :h :i :j :k :l)))
