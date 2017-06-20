Usage of Baxter
===============

How to control baxter via roseus

preparation
-----------

  Run below under emacs's shell environment (`M-x shell`).

  .. code-block:: bash

    roseus

  When you start new shell, DO NOT FORGET to run:

  .. code-block:: bash

    rossetip
    rossetmaster baxter
    source ~/catkin_ws/devel/setup.bash

  Set baxter-interface

  .. code-block:: lisp

    ;;load modules
    (load "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l")
    ;;create a robot model(*baxter*) and make connection to the real robot(*ri*)
    (jsk_arc2017_baxter::arc-init)
    ;; display the robot model
    (objects (list *baxter*))


arc-interface function APIs
---------------------------

- rotate left(right) gripper

  .. code-block:: lisp

    (send *baxter* :rotate-gripper :larm 90 :relative nil)


- slide right gripper

  .. code-block:: lisp

    (send *baxter* :slide-gripper :rarm 50 :relative nil)


- move fingers in right gripper

  .. code-block:: lisp

    (send *baxter* :hand :rarm :angle-vector #f(90 90))
    (send *baxter* :hand-grasp-pre-pose :rarm :opposed)
    (send *baxter* :hand-grasp-pose :rarm :cylindrical)


- send initial pose for arc2017

  .. code-block:: lisp

    (send *baxter* :fold-pose-back)


- send current joint angles of robot model to real robot

  .. code-block:: lisp

    (send *ti* :send-av)


- send current hand joint angles of robot model to real robot

  .. code-block:: lisp

    (send *ri* :move-hand :rarm (send *baxter* :hand :rarm :angle-vector) 1000)


