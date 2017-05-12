jsk_arc2017_baxter
================

see https://github.com/jsk-ros-pkg/jsk_robot/blob/master/jsk_fetch_robot/README.md

control robot via roseus
------------------------

run `roseus` under emacs's shell environment (`M-x shell`), when you start new shell, do not forget to run `rossetip`, `rossetmaster baxter` and `source ~/catkin_ws/devel/setup.bash`

```
(load "package://jsk_arc2017_baxter/euslisp/lib/arc-interface.l") ;; load modules
(jsk_arc2017_baxter::arc-init) ;;create a robot model(*baxter*) and make connection to the real robot(*ri*)
(objects (list *baxter*))        ;; display the robot model
```

arc-interface function APIs
-----------------------------

- rotate left(right) gripper

```
(send *baxter* :rotate-gripper :larm 90)
```

- send initial pose for arc2017

```
(send *baxter* :fold-pose-back)
```

- send current joint angles of robot model to real robot

```
(send *ri* :send-av)
```
