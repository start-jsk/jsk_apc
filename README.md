2014-semi
=========

[![CI Status at https://travis-ci.org/start-jsk/2014-semi/](https://travis-ci.org/start-jsk/2014-semi.svg)](https://travis-ci.org/start-jsk/2014-semi)
[![Slack](https://img.shields.io/badge/slack-jsk--seminar--2014-blue.svg)](https://jsk-seminar-2014.slack.com)

2014 機械工学少人数ゼミ　プロジェクトページ

環境の構築
----------
[Ros Wiki](http://wiki.ros.org/indigo/Installation/Ubuntu)のインストール手順に従ってindigoの環境設定をしたのち,
以下の手順で環境構築をしてください.
```
$ mkdir -p catkin_ws/semi/src
$ cd catkin_ws/semi/src
$ wstool init
$ wstool merge https://raw.githubusercontent.com/start-jsk/2014-semi/master/jsk_2014_picking_challenge.rosinstall
$ wstool update
$ cd ..
$ rosdep install -y -r --from-paths .
$ sudo apt-get install python-catkin-tools ros-indigo-jsk-tools
$ catkin build
$ source devel/setup.bash
```

実機を使うときの環境設定
-----------------------
```
$ rossetmaster baxter.jsk.imi.i.u-tokyo.ac.jp
$ rossetip
$ sudo ntpdate baxter.jsk.imi.i.u-tokyo.ac.jp
```

euslispからロボットを動かす
--------------------------
```
> rosrun jsk_2014_picking_challenge main.l
$ (test-1) ;; simple example
$ (test-2) ;; ik exmple
```

最新のデモの実行方法
-----------------------

```sh
roslaunch jsk_baxter_startup baxter.launch
roslaunch jsk_2014_picking_challenge challenge.launch
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

Kinect2のlocalセットアップについて
----------------------------------
https://github.com/code-iai/iai_kinect2 のinstall手順に従い
インストールしてください。
