#!/usr/bin/env python

import os.path as osp

import chainer
import fcn
import numpy as np
import tqdm

import instance_occlsegm_lib


model = fcn.models.FCN8sAtOnce(n_class=2)
model_file = 'logs/train_fcn_fgbg/20180116_202338/model_00010000.npz'
chainer.serializers.load_npz(model_file, model)

gpu = 0
chainer.cuda.get_device_from_id(gpu).use()
model.to_gpu()

obj_data_dir = osp.expanduser('~/data/arc2017/datasets/ItemDataARC2017')
obj_names, obj_data = instance_occlsegm_lib.datasets.apc.arc2017.\
    load_item_data(obj_data_dir, skip_known=False)

for obj_datum in tqdm.tqdm(obj_data):
    img_file = obj_datum['img_file']
    img = obj_datum['img']
    lbl_fgbg = obj_datum['mask'].astype(np.int32)

    x = fcn.datasets.transform_lsvrc2012_vgg16((img, lbl_fgbg))[0]
    x = chainer.cuda.to_gpu(x)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        model(x[None])
        lbl_pred = model.xp.argmax(model.score.array, axis=1)
    lbl_pred = chainer.cuda.to_cpu(lbl_pred)[0]
    # lbl_fgbg = instance_occlsegm_lib.image.label2rgb(lbl_fgbg, img)
    # lbl_pred = instance_occlsegm_lib.image.label2rgb(lbl_pred, img)
    # instance_occlsegm_lib.io.imshow(instance_occlsegm_lib.image.tile(
    #    [img, lbl_fgbg, lbl_pred]))
    # instance_occlsegm_lib.io.waitkey()

    mask_pred = lbl_pred.astype(bool)
    mask_file = img_file + '.mask_pred.npz'
    np.savez_compressed(mask_file, mask_pred)
