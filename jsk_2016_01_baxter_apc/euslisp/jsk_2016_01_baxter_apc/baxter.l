;; vim: set ft=lisp:

(require "package://baxtereus/baxter-util.l")
(require "package://jsk_2016_01_baxter_apc/robots/baxter.l")

(unless (find-package "JSK_2016_01_BAXTER_APC")
  (make-package "JSK_2016_01_BAXTER_APC"))

(defclass jsk_2016_01_baxter_apc::baxter-robot
  :super baxter-robot
  :slots (_view-hand-pos _ik-prepared-poses))
(defmethod jsk_2016_01_baxter_apc::baxter-robot
  (:init
    (&rest args)
    (send-super* :init args)
    ;; initialize _view-hand-pos
    (setq _view-hand-pos (make-hash-table))
    (sethash :rarm _view-hand-pos (make-hash-table))
    (sethash :larm _view-hand-pos (make-hash-table))
    ;; for larm
    (sethash :a (gethash :larm _view-hand-pos) #f(785.344 229.224 762.48))
    (sethash :b (gethash :larm _view-hand-pos) #f(813.742 10.946 828.431))
    (sethash :d (gethash :larm _view-hand-pos) #f(807.059 275.852 633.668))
    (sethash :e (gethash :larm _view-hand-pos) #f(770.657 57.875 550.976))
    (sethash :g (gethash :larm _view-hand-pos) #f(781.985 244.363 368.102))
    (sethash :h (gethash :larm _view-hand-pos) #f(819.589 13.426 415.42))
    (sethash :j (gethash :larm _view-hand-pos) #f(811.875 312.473 156.184))
    (sethash :k (gethash :larm _view-hand-pos) #f(780.791 15.733 137.103))
    ;; for rarm
    (sethash :b (gethash :rarm _view-hand-pos) #f(813.742 -10.946 828.431))
    (sethash :c (gethash :rarm _view-hand-pos) #f(785.344 -180 762.48))
    (sethash :e (gethash :rarm _view-hand-pos) #f(770.657 -57.875 550.976))
    (sethash :f (gethash :rarm _view-hand-pos) #f(807.059 -186 633.668))
    (sethash :h (gethash :rarm _view-hand-pos) #f(819.589 -13.426 415.42))
    (sethash :i (gethash :rarm _view-hand-pos) #f(781.985 -184 368.102))
    (sethash :k (gethash :rarm _view-hand-pos) #f(780.791 -15.733 137.103))
    (sethash :l (gethash :rarm _view-hand-pos) #f(811.875 -180 156.184))
    ;; pos of :c, :f, :i, :l is not symmetrical to :a, :d, :g, :j because torso can't see
    ;; initialize _ik-prepared-poses
    (setq _ik-prepared-poses (list :untuck-pose))
    )
  (:set-ik-prepared-poses
    (poses)
    (if (listp poses)
      (setq _ik-prepared-poses poses))
    )
  (:ik-prepared-poses () _ik-prepared-poses)
  (:inverse-kinematics
    (target-coords
      &rest args
      &key move-target
      (link-list
        (if (atom move-target)
          (send self :link-list (send move-target :parent))
          (mapcar #'(lambda (mt) (send self :link-list (send mt :parent))) move-target)))
      (use-gripper nil) (rthre (deg2rad 10))
      &allow-other-keys)
    ;; if the last link of link-list is a gripper, remove them
    (if (null use-gripper)
      (when (member (send (car (last link-list)) :name)
                    (list "right_gripper_vacuum_pad_base"
                          "left_gripper_vacuum_pad_base") :test #'equal)
        (setq link-list (butlast link-list))) nil)
    (send-super* :inverse-kinematics target-coords
                 :move-target move-target :link-list link-list :rthre rthre args)
    )
  (:rotate-wrist
    (arm angle &key (relative t))
    (send self arm :wrist-r :joint-angle angle :relative relative)
    (send self :angle-vector)
    )
  (:rotate-gripper
    (arm angle &key (relative t))
    (send self arm :gripper-p :joint-angle angle :relative relative)
    (send self :angle-vector)
    )
  (:avoid-shelf-pose
    (arm bin)
    (case bin
      (:a (send self :avoid-shelf-pose-a arm))
      (:b (send self :avoid-shelf-pose-b arm))
      (:c (send self :avoid-shelf-pose-c arm))
      (:d (send self :avoid-shelf-pose-d arm))
      (:e (send self :avoid-shelf-pose-e arm))
      (:f (send self :avoid-shelf-pose-f arm))
      (:g (send self :avoid-shelf-pose-g arm))
      (:h (send self :avoid-shelf-pose-h arm))
      (:i (send self :avoid-shelf-pose-i arm))
      (:j (send self :avoid-shelf-pose-j arm))
      (:k (send self :avoid-shelf-pose-k arm))
      (:l (send self :avoid-shelf-pose-l arm)))
    )
  (:view-hand-pose
    (arm bin)
    (case bin
      (:a (send self :view-hand-pose-a arm))
      (:b (send self :view-hand-pose-b arm))
      (:c (send self :view-hand-pose-c arm))
      (:d (send self :view-hand-pose-d arm))
      (:e (send self :view-hand-pose-e arm))
      (:f (send self :view-hand-pose-f arm))
      (:g (send self :view-hand-pose-g arm))
      (:h (send self :view-hand-pose-h arm))
      (:i (send self :view-hand-pose-i arm))
      (:j (send self :view-hand-pose-j arm))
      (:k (send self :view-hand-pose-k arm))
      (:l (send self :view-hand-pose-l arm)))
    (let ((pos (gethash bin (gethash arm _view-hand-pos))))
      (if pos
        (send self arm :inverse-kinematics
              (make-coords :pos pos)
              :rotation-axis nil
              :revert-if-fail nil)
        )
      )
    )
  (:visualize-path
    (avs &key (limb))
    (let (robots coords-list)
      (dotimes (i (length avs))
        (let ((robot (instance (class self) :init)))
          (send robot :angle-vector (elt avs i))
          (when (not (eq i (1- (length avs))))
            (send-all (send robot :bodies) :set-color nil 0.3)
            )
          (pushback robot robots)
          (when limb
            (pushback (send robot limb :end-coords) coords-list)
            )
          )
        )
      (objects (append robots coords-list))
      )
    )
  (:view-opposite-hand-pose
    (arm &key (dy 300))
    (let (opposite-arm gripper-coords theta)
      (setq opposite-arm (if (eq arm :larm) :rarm :larm))
      (setq gripper-coords
            (send (send self opposite-arm :end-coords)
                  :copy-worldcoords))
      (send gripper-coords :orient 0 :z :world)
      (if (eq arm :larm)
        (setq theta -1.57)
        (progn
          (setq dy (- dy))
          (setq theta 1.57)))
      (send gripper-coords :translate (float-vector 0 dy -200))
      (send gripper-coords :rotate theta :z :world)
      (send self :rotate-gripper arm 70 :relative nil)
      (send self arm
            :inverse-kinematics gripper-coords
            :use-gripper nil
            :rotation-axis t
            :revert-if-fail nil)))
  (:get-support-arm-coords
    (arm &key (dx 0) (dy 0) (dz 0))
    (let (support-arm-coords)
      (setq support-arm-coords
            (send (send self arm :end-coords)
                  :copy-worldcoords))
      (send support-arm-coords :orient 0 :z :world)
      (send support-arm-coords :rotate 3.14 :y :world)
      (send support-arm-coords :translate (float-vector dx dy dz) :world)
      support-arm-coords))
  (:ik->nearest-pose
    (arm target-coords pose-candidates &rest args)
    (let ((opposite-arm (if (eq arm :larm) :rarm :larm))
          current-opposite-av current-av chosen-pose)
      (unless pose-candidates
        (setq pose-candidates (list (send self :angle-vector))))
      (let (current-opposite-av current-av ik-solvable-poses chosen-pose)
        (setq current-opposite-av (send self opposite-arm :angle-vector))
        (setq current-av (send self arm :angle-vector))
        ;; exclude poses which IK fail from
        (dolist (pose pose-candidates)
          (send self :angle-vector pose)
          (if (send* self arm :inverse-kinematics target-coords args)
            (pushback pose ik-solvable-poses)))
        (unless ik-solvable-poses
          (ros::ros-error "[:ik->nearest-pose] Cannot solve IK from poses")
          (send self opposite-arm :angle-vector current-opposite-av)
          (send self arm :angle-vector current-av)
          (return-from :ik->nearest-pose nil))
        (setq chosen-pose
              (car (sort ik-solvable-poses #'<
                         #'(lambda (pose)
                             (norm
                               (v-
                                 current-av
                                 (progn
                                   (send self :angle-vector pose)
                                   (send* self arm :inverse-kinematics target-coords args)
                                   (send self arm :angle-vector))))))))
        (format t "[:ik->nearest-pose] arm:~a midpose: ~a~%" arm chosen-pose)
        (send self :angle-vector chosen-pose)
        (send self opposite-arm :angle-vector current-opposite-av)
        (send* self arm :inverse-kinematics target-coords args))))
  )
