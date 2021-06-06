# `occluded_grasp_mask_rcnn`: Mask-RCNN with occlusion recognition and dual-arm grasping point detection

## Training

```bash
# single GPU
python train.py --gpu 0
# multi GPU 
mpiexec -n 4 python train.py --multi-node

```

## Demo

```bash
python demo_test.py --gpu 0 --logs logs/20181108_020501.794412 --pretrained-model logs/20181108_020501.794412/OccludedGraspMaskRCNNResNet101_model_iter_26934.npz
```
