2014-semi
=========

2014 機械工学少人数ゼミ　プロジェクトページ

環境の構築
----------
```
$ sudo apt-get install julius julius-dev
$ sudo apt-get install sudo apt-get install libzbar0 libzbar-dev
$ mkdir -p catkin_ws/semi/src
$ cd  catkin_ws/semi/src
$ wstool init
$ wstool set 2014-semi https://github.com/start-jsk/2014-semi --git
$ wstool set jsk_robot https://github.com/jsk-ros-pkg/jsk_robot --git
$ wstool set zbar_ros git@github.com:vicoslab/vicos_ros.git
$ wstool update
$ cd ..
$ rosdep install -y -r --from-paths .
$ catkin_make
$ source devel/setup.bash
```


実機を使うときの環境設定
-----------------------
```
$ source `rospack find jsk_tools`/src/bashrc.ros
$ rossetrobot baxter.jsk.imi.i.u-tokyo.ac.jp
$ rossetip
$ sudo ntpdate baxter.jsk.imi.i.u-tokyo.ac.jp
```

euslispからロボットを動かす
--------------------------
```
> roscd jsk_2014_picking_challenge
> roseus scripts/main.l
$ (init)
$ (test-1) ;; simple example
$ (test-2) ;; ik exmple
```

rvizで今の状態を表示する
------------------------

```
$ rosrun rviz rviz
```

launchファイルを使う
-----------------------

```
$ roslaunch jsk_2014_picking_challenge baxter_oneclick_grasp.launch
```

* baxter_oneclick_grasp.launch

rviz上で表示されているboxをクリックをすることでつかみにいきます
このlaunchではbaxter_organized_multi.launchをincludeしています。

* baxter_organized_multi.launch

平面と平面上の物体を分離し,平面とBounding Boxをpublishします。
jsk_pcl_rosのorganized_multi_planeのlaunchをincludeしています。


addを選択しRobotModelを追加, FixedFrame base に設定
