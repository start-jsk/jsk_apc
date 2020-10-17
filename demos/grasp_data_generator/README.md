# Automatic dataset generation and training for dual-arm grasping 

## Installation

```bash
pip install --user -e .
```

## Advanced Robotics 2020: Selective dual-arm occluded grasping

### Generating datasets

Read [scripts/README.md](./scripts/README.md)


### Training OccludedGraspMaskRCNN with datasets

Read [experiments/occluded_grasp_mask_rcnn/README.md](./experiments/occluded_grasp_mask_rcnn/README.md)

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

## IROS2018: Selective dual-arm grasping

### Generating datasets

Read [scripts/README.md](./scripts/README.md)


### Training DualarmGraspFCN with datasets

Read [experiments/dualarm_grasp_fcn/README.md](./experiments/dualarm_grasp_fcn/README.md)

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
