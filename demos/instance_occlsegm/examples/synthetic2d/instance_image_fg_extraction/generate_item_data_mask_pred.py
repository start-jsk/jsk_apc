#!/usr/bin/env python

import argparse

import chainer
import fcn
import numpy as np
import tqdm

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import synthetic2d


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    item_data_choices = list(
        synthetic2d.datasets.InstanceImageDataset.item_data_dirs.keys()
    )
    parser.add_argument(
        '--item-data',
        default='arc2017_all',
        choices=item_data_choices,
        help='item data',
    )
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    model = fcn.models.FCN8sAtOnce(n_class=2)
    model_file = instance_occlsegm_lib.data.download(
        url='https://drive.google.com/uc?id=1k7qCONNta6WDjqdXAXdem8rDQjWPWeFb',
        md5='5739cb23249993428dcc67e6d763b00a',
    )
    chainer.serializers.load_npz(model_file, model)

    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

    dataset = synthetic2d.datasets.InstanceImageDataset(
        item_data_dir=args.item_data, load_pred=False
    )
    obj_data_dir = dataset.item_data_dir
    obj_names, obj_data = instance_occlsegm_lib.datasets.apc.\
        arc2017.load_item_data(obj_data_dir, skip_known=False)

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
        # instance_occlsegm_lib.io.imshow(
        #    instance_occlsegm_lib.image.tile([img, lbl_fgbg, lbl_pred]))
        # instance_occlsegm_lib.io.waitkey()

        mask_pred = lbl_pred.astype(np.uint8) * 255
        mask_file = img_file + '.mask_pred.jpg'
        instance_occlsegm_lib.io.imsave(mask_file, mask_pred)


if __name__ == '__main__':
    main()
