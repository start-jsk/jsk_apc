# Advanced Robotics 2020: Selective dual-arm occluded grasping

## Sampling 

### Setup

```bash
roslaunch dualarm_grasping baxter.launch
roslaunch dualarm_grasping setup_occluded_sampling.launch
```

Pass args for `first_sampling:=true` or `second_sampling:=true`

### Main Program

```bash
roseus euslisp/sampling-grasp.l
```

## Execution (Cluster Grasping)

### Setup

```bash
roslaunch dualarm_grasping baxter.launch
roslaunch dualarm_grasping setup_occluded_picking.launch
```

### Main Program

```bash
roseus euslisp/dualarm-grasp.l
```

## Execution (Target Grasping)

### Setup

```bash
roslaunch dualarm_grasping baxter.launch
roslaunch dualarm_grasping setup_occluded_picking.launch target_grasp:=true
```

### Main Program

```bash
roseus euslisp/dualarm-grasp.l
```

## Evaluation (Target Grasping)

### Setup

```bash
roslaunch dualarm_grasping baxter.launch
roslaunch dualarm_grasping setup_occluded_test.launch target_grasp:=true
```

### Main Program

```bash
roseus euslisp/test-grasp.l
```

## Citation

```bib
@article{doi:10.1080/01691864.2020.1783352,
  author = {Shingo Kitagawa and Kentaro Wada and Shun Hasegawa and Kei Okada and Masayuki Inaba},
  title = {Few-experiential learning system of robotic picking task with selective dual-arm grasping},
  journal = {Advanced Robotics},
  volume = {0},
  number = {0},
  pages = {1-19},
  year  = {2020},
  publisher = {Taylor & Francis},
  doi = {10.1080/01691864.2020.1783352},
  URL = {https://doi.org/10.1080/01691864.2020.1783352},
  eprint = {https://doi.org/10.1080/01691864.2020.1783352}
}
```

# IROS2018: Selective dual-arm grasping

## Sampling 

### Setup

```bash
roslaunch dualarm_grasping baxter.launch
roslaunch dualarm_grasping setup_sampling.launch
```

Pass args for `first_sampling:=true` or `second_sampling:=true`

### Main Program

```bash
roseus euslisp/sampling-grasp.l
```

## Execution

### Setup

```bash
roslaunch dualarm_grasping baxter.launch
roslaunch dualarm_grasping setup_picking.launch
```

### Main Program

```bash
roseus euslisp/dualarm-grasp.l
```

## Evaluation 

### Setup

```bash
roslaunch dualarm_grasping baxter.launch
roslaunch dualarm_grasping setup_test.launch
```

### Main Program

```bash
roseus euslisp/test-grasp.l
```

## Citation

```bib
@inproceedings{SelectiveDualarmGrasping:IROS2018,
  author={Shingo Kitagawa and Kentaro Wada and Shun Hasegawa and Kei Okada and Masayuki Inaba},
  booktitle={Proceedings of The 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  title={Multi-stage Learning of Selective Dual-arm Grasping Based on Obtaining and Pruning Grasping Points Through the Robot Experience in the Real World},
  year={2018},
  month={october},
  pages={7123-7130},
}
```
