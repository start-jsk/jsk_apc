;; vim: set ft=lisp:


(defmacro pushback (el lst)
  `(if (null ,lst)
       (setf ,lst (list ,el))
     (nconc ,lst (list ,el))))


(defun m->mm (m) (* 1000 m))


(defun argmax (fvec)
  (let (ret val maxv)
    (dotimes (i (length fvec))
      (setq val (elt fvec i))
      (cond ((= i 0) (setq maxv val ret i))
            ((> val maxv) (setq maxv val ret i))
            (t nil)))
    ret))

(defun argmin (fvec)
  (let (ret val minv)
    (dotimes (i (length fvec))
      (setq val (elt fvec i))
      (cond ((= i 0) (setq minv val ret i))
            ((> minv val) (setq minv val ret i))
            (t nil)))
    ret))


(defun vec-list-max (vec-list &key axis)
  (let (vec max-vec)
    (dotimes (i (length vec-list))
      (setq vec (elt vec-list i))
      (cond
        ((= i 0) (setq max-vec vec))
        ((< (elt max-vec axis) (elt vec axis)) (setq max-vec vec))
        (t nil)
        )
      )
    max-vec))


(defun str2symbol (str)
  (intern (string-upcase str) "KEYWORD"))


(defun symbol2str (_symbol)
  (string-downcase (symbol-string _symbol)))


(defun ros::advertise-if-yet (name data-class queue-size)
  (unless (ros::get-topic-publisher name)
    (ros::advertise name data-class queue-size)
    (unix::sleep 1)))


(defun underscore-to-space (str_)
  (let* ((str (copy-list str_)))
    (while
      (position #\_ str)
      (setf (schar str (position #\_ str)) #\ ))
    str))

(defun which-bin-region (bin)
  (let ((l-bins '(:a :d :g :j))
        (c-bins '(:b :e :h :k))
        (r-bins '(:c :f :i :l)))
    (cond
      ((member bin l-bins) :left)
      ((member bin c-bins) :center)
      ((member bin r-bins) :right)
      (t nil))))


(defun arm-to-ctype (arm)
  (case arm
    (:larm :larm-controller)
    (:rarm :rarm-controller)))


(defun arm2str (arm) (arm-to-str arm))
(defun arm-to-str (arm)
  (cond
    ((eq arm :larm) "left")
    ((eq arm :rarm) "right")
    (t nil)))

(defun opposite-arm (arm)
  (cond
    ((eq arm :larm) :rarm)
    ((eq arm :rarm) :larm)
    (t nil)))


(defun get-object-size (object-name)
  (let (table)
    (setq table (make-hash-table))
    (sethash (str2symbol "champion_copper_plus_spark_plug") table 0.105650)
    (sethash (str2symbol "rolodex_jumbo_pencil_cup") table 0.131390)
    (sethash (str2symbol "elmers_washable_no_run_school_glue") table 0.140944)
    (sethash (str2symbol "sharpie_accent_tank_style_highlighters") table 0.182510)
    (sethash (str2symbol "highland_6539_self_stick_notes") table 0.186724)
    (sethash (str2symbol "paper_mate_12_count_mirado_black_warrior") table 0.202294)
    (sethash (str2symbol "mommys_helper_outlet_plugs") table 0.225713)
    (sethash (str2symbol "laugh_out_loud_joke_book") table 0.267417)
    (sethash (str2symbol "expo_dry_erase_board_eraser") table 0.276404)
    (sethash (str2symbol "mead_index_cards") table 0.319240)
    (sethash (str2symbol "munchkin_white_hot_duck_bath_toy") table 0.335600)
    (sethash (str2symbol "mark_twain_huckleberry_finn") table 0.363938)
    (sethash (str2symbol "feline_greenies_dental_treats") table 0.385943)
    (sethash (str2symbol "dr_browns_bottle_brush") table 0.412830)
    (sethash (str2symbol "crayola_64_ct") table 0.417038)
    (sethash (str2symbol "safety_works_safety_glasses") table 0.570724)
    (sethash (str2symbol "stanley_66_052") table 0.577224)
    (sethash (str2symbol "first_years_take_and_toss_straw_cup") table 0.584461)
    (sethash (str2symbol "kong_air_dog_squeakair_tennis_ball") table 0.597475)
    (sethash (str2symbol "kong_duck_dog_toy") table 0.611756)
    (sethash (str2symbol "oreo_mega_stuf") table 0.631798)
    (sethash (str2symbol "kyjen_squeakin_eggs_plush_puppies") table 0.637863)
    (sethash (str2symbol "kong_sitting_frog_dog_toy") table 0.701482)
    (sethash (str2symbol "genuine_joe_plastic_stir_sticks") table 0.727586)
    (sethash (str2symbol "cheezit_big_original") table 1.000000)
  (gethash (str2symbol object-name) table)))


;; >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
;; For Python Users
;; >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
(defun zip (a b) (mapcar #'cons a b))

(defun dict (zipped)
  (let ((table (make-hash-table :test #'equal)))
    (dolist (el zipped)
      (sethash (car el) table (cdr el))
      )
    table))
