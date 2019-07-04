# Selective Dualarm Stowing 

## Paper
Shingo Kitagawa, Kentaro Wada, Kei Okada, Masayuki Inaba:<br>
Learning-based Task Failure Prediction for Selective Dual-arm Manipulation in Warehouse Stowing,<br>
in The 15th International Conference on Intellignet Autonomous Systems, 2018.

## Prerequisition

- jsk_apc
- Chainer

## How to run
### Data Collection

```bash
$ roslaunch jsk_2016_01_baxter_apc baxter.launch moveit:=true
# other pane
$ roslaunch jsk_2016_01_baxter_apc setup_for_stow.launch
# other pane
$ roslaunch selective_dualarm_stowing bimanual_stow_data_collection.launch
# other pane
$ rosrun selective_dualarm_stowing bimanual-stow-data-collection.l
```
### Training
```
$ cd experiments
$ ./train_alex.py -h
usage: train_alex.py [-h] [--gpu GPU] [-o OUT] [--resume RESUME] [--loop LOOP]
                     [--threshold THRESHOLD] [--test-size TEST_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU
  -o OUT, --out OUT
  --resume RESUME
  --loop LOOP
  --threshold THRESHOLD
  --test-size TEST_SIZE
# Example
$ ./train_alex.py --gpu 0 -o ../logs/alex/latest --loop 10 --threshold 0.5 --test-size 0.2
```

### Execution

```bash
$ roslaunch jsk_2016_01_baxter_apc baxter.launch moveit:=true
# other pane
$ roslaunch jsk_2016_01_baxter_apc setup_for_stow.launch
# other pane
$ roslaunch selective_dualarm_stowing selective_dualarm_stowing.launch
# other pane
$ rosrun selective_dualarm_stowing selective-stow.l
```

## Citation

```
@inproceedings{SelectiveDualarmStowing:Kitagawa:IAS15,
  author={Shingo Kitagawa and Kentaro Wada and Kei Okada and Masayuki Inaba},
  booktitle={The 15th International Conference on Intellignet Autonomous Systems},
  title={Learning-based Task Failure Prediction for Selective Dual-arm Manipulation in Warehouse Stowing},
  year={2018},
  month={june}
}
```

