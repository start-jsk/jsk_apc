2014-semi
=========

2014 機械工学少人数ゼミ　プロジェクトページ


環境の構築
----------
```
mkdir -p catkin_ws/semi/src
cd  catkin_ws/semi/src
wstool init
wstool git set 2014-semi https://github.com/start-jsk/2014-semi --git 
wstool update
cd ..
catkin_make
source devel/setup.bash
```


実機を使うときの環境設定
-----------------------
```
source `rospack find jsk_tools`/src/bashrc.ros
rossetrobot baxter.jsk.imi.i.u-tokyo.ac.jp
rossetip
```

euslispからロボットを動かす
--------------------------
```
> roscd 2014_picking_challenge
> roseus scripts/main.l
$ (init)
$ (main)
```


