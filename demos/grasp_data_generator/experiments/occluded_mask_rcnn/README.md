# `occluded_mask_rcnn`: Mask-RCNN with occlusion recognition 

## Training

```bash
# single GPU
python train.py --gpu 0
# multi GPU 
mpiexec -n 4 python train.py --multi-node

```

## Demo

```bash
python demo_test.py --gpu 0 --logs logs/20181107_032434.121838 --pretrained-model logs/20181107_032434.121838/OccludedMaskRCNNResNet_model_iter_26934.npz
```
