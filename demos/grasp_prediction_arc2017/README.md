# grasp_prediction_arc2017


- Object label and grasp affordance segmentation learned with instance image stacking
- Picking folded objects (e.g., books) with the Suction Pinching Hand

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
# Don't do the following with soursing ROS setup.*sh
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
# Or if you want to use Anaconda environment:
# source .anaconda/bin/activate  # Don't do this with soursing ROS setup.*sh
```

##### Training CNNs

###### Requirements

- cupy
  ```bash
  pip install cupy-cuda92  # or cupy, cupy-cuda80, cupy-cuda90, ...
  ```

###### Object label and grasp affordance segmentation with ARC2017 dataset

```bash
cd examples/grasp_prediction_arc2017
./train_fcn32s.py -g 0 -d -p wada_icra2018
```

###### Object label and grasp affordance segmentation with book dataset

```bash
cd examples/grasp_prediction_arc2017
./train_fcn8s.py -g 0 -d -p hasegawa_iros2018  # or hasegawa_mthesis
```


### With ROS

```bash
roslaunch grasp_prediction_arc2017 sample_fcn_object_segmentation.launch
```
![](ros/grasp_prediction_arc2017/samples/images/fcn_object_segmentation.jpg)


## ARC2017 demonstration

In [execution flow of pick task imitating ARC2017 competition](https://jsk-apc.readthedocs.io/en/latest/jsk_arc2017_baxter/arc2017_pick_trial.html#with-environment-imitating-arc2017-pick-competition), execute

```bash
roslaunch grasp_prediction_arc2017 setup_for_pick.launch
```

instead of

```bash
roslaunch jsk_arc2017_baxter setup_for_pick.launch
```

### Video (Click Below)

<div align="center">
  <a href="https://drive.google.com/uc?id=1uf-zMi3m2YtnAub4POBR8EAiStW7QDkv">
    <img src="https://drive.google.com/uc?export=view&id=1xS8fuoIn_dhBCr5xd9BIjtDbqwyQxl6s" />
  </a>
</div>


## Hasegawa IROS2018 Demo: Pick and Insert Books

### Setup

```bash
rosrun grasp_prediction_arc2017 install_hasegawa_iros2018
```

### Execution

```bash
roslaunch grasp_prediction_arc2017 baxterlgv7.launch
roslaunch grasp_prediction_arc2017 setup_for_book_picking.launch hasegawa_iros2018:=true
roslaunch grasp_prediction_arc2017 book_picking.launch json_dir:=`rospack find grasp_prediction_arc2017`/json_dirs/hasegawa_iros2018/ForItemDataBooks6/layout1
```

### Video (Click Below)

<div align="center">
  <a href="https://drive.google.com/uc?id=1MBwzwkSWH23wujnzDtNFKRULViJP-ZEy">
    <img src="https://drive.google.com/uc?export=view&id=1lEVKdUM9_08XlVqKk-OStBnb-hpNcwgN" />
  </a>
</div>

### Citation

```
@INPROCEEDINGS{hasegawa2018detecting,
  author={S. {Hasegawa} and K. {Wada} and K. {Okada} and M. {Inaba}},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  title={Detecting and Picking of Folded Objects with a Multiple Sensor Integrated Robot Hand},
  year={2018},
  pages={1138-1145},
  doi={10.1109/IROS.2018.8593398},
  ISSN={2153-0866},
  month={Oct.}
}
```


## Hasegawa Master Thesis Demo: Grasp Books with Low Lifting

### Setup

```bash
rosrun grasp_prediction_arc2017 install_hasegawa_mthesis
```

### Execution

```bash
# Experiments of Grasp Stability
roslaunch grasp_prediction_arc2017 baxterlgv7.launch
roslaunch grasp_prediction_arc2017 setup_for_book_picking.launch hasegawa_mthesis:=true
roslaunch grasp_prediction_arc2017 book_picking.launch main:=false json_dir:=`rospack find grasp_prediction_arc2017`/json_dirs/hasegawa_mthesis/ForItemDataBooks8/each_obj/alpha_cubic_sport_wallet
roseus `rospack find grasp_prediction_arc2017`/euslisp/hasegawa_mthesis/pick-book-eval.l

# In Euslisp Interpreter
(pick-book-eval-init :ctype :larm-head-controller :moveit t)
(pick-book-eval-mainloop :larm)
## Please see warn messages and source codes for optional settings
```
