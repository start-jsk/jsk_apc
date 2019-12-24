# grasp_prediction_arc2017


## Installation

First, [install jsk_apc](https://github.com/start-jsk/jsk_apc#installation).

```bash
source ~/ros/ws_jsk_apc/devel/setup.bash  # source jsk_apc workspace

WS=~/ros_mvtk/src
mkdir -p $WS
cd $WS

wstool init
wstool set ros_mvtk https://github.com/wkentaro/ros_mvtk.git --git -vmaster -y -u

cd ..
catkin build
```


## Nodes

### fcn_object_segmentation.py

```bash
roslaunch sample_fcn_object_segmentation.launch
```

![](samples/images/fcn_object_segmentation.jpg)


## ARC2017 demonstration

```bash
roslaunch jsk_arc2017_baxter baxter.launch
roslaunch grasp_prediction_arc2017 setup_for_pick.launch
roslaunch jsk_arc2017_baxter pick.launch json_dir:=<json_dir>
```


## Hasegawa IROS2018 Demo: Pick and Insert Books

```bash
roscd jsk_apc
git remote add pazeshun https://github.com/pazeshun/jsk_apc.git
git fetch pazeshun
git checkout baxterlgv7-book-picking
catkin build

roslaunch jsk_arc2017_baxter baxterlgv7.launch book_picking:=true
roslaunch grasp_prediction_arc2017 setup_for_pick_baxterlgv7.launch
roslaunch jsk_arc2017_baxter pick_book.launch json_dir:=<json_dir>
```
