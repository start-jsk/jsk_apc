# grasp_prediction_arc2017


## Installation

### With ROS

[Install jsk_apc](https://github.com/start-jsk/jsk_apc#installation).

### (At Your Own Risk) Without ROS (Use Anaconda)

```bash
make install  # Python3
# make install2  # Python2
```

##### Usage

```bash
source .anaconda/bin/activate
python -c 'import grasp_prediction_arc2017_lib'
```

##### Testing

```bash
make lint
```


## Examples

### Without ROS

First, you must activate python environment:
```bash
source ~/ros/ws_jsk_apc/devel/.private/grasp_prediction_arc2017/share/grasp_prediction_arc2017/venv/bin/activate
# If you want to use Anaconda environment:
# source .anaconda/bin/activate
```

##### Training CNNs

###### Requirements

- cupy
  ```bash
  pip install cupy-cuda92  # cupy, cupy-cuda80, cupy-cuda90, or ...
  ```

###### Object label and grasp affordance segmentation with ARC2017 dataset

```bash
cd examples/grasp_prediction_arc2017
./train_fcn32s.py -g 0 -d -p wada_icra2018
```

###### Object label and grasp affordance segmentation with book dataset (Hasegawa IROS2018)

```bash
cd examples/grasp_prediction_arc2017
./train_fcn8s.py -g 0 -d -p hasegawa_iros2018
```


### With ROS

```bash
roslaunch grasp_prediction_arc2017 sample_fcn_object_segmentation.launch
```
![](ros/grasp_prediction_arc2017/samples/images/fcn_object_segmentation.jpg)


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
