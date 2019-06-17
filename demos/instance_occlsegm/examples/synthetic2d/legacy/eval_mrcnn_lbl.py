#!/usr/bin/env python

import argparse
import os.path as osp
import pprint

import chainer
import yaml

import chainer_mask_rcnn as mrcnn

import contrib

from train_mrcnn_lbl import MaskRcnnDataset


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    gpu = args.gpu
    log_dir = args.log_dir

    chainer.cuda.get_device_from_id(gpu).use()

    with open(osp.join(log_dir, 'params.yaml')) as f:
        params = yaml.load(f)

    # dataset
    test_data = contrib.datasets.ARC2017OcclusionDataset('test')
    class_names = test_data.class_names[1:]
    test_data_list = test_data.get_video_datasets()
    del test_data
    test_data_list = [MaskRcnnDataset(td) for td in test_data_list]

    # model
    mask_rcnn = contrib.models.MaskRCNNResNet(
        n_layers=int(params['model'].lstrip('resnet')),
        n_fg_class=len(class_names),
        pretrained_model=osp.join(log_dir, 'snapshot_model.npz'),
        min_size=params['min_size'],
        max_size=params['max_size'],
        anchor_scales=params['anchor_scales'],
        mask_loss=params['mask_loss'],
        pooling_func=mrcnn.functions.roi_align_2d,
    )
    mask_rcnn.to_gpu()

    # iterator
    test_data_list = [
        chainer.datasets.TransformDataset(
            td,
            mrcnn.datasets.MaskRCNNTransform(
                mask_rcnn,
                train=False,
            ),
        )
        for td in test_data_list
    ]
    test_iters = {
        i: chainer.iterators.SerialIterator(
            td,
            batch_size=1,
            repeat=False,
            shuffle=False,
        )
        for i, td in enumerate(test_data_list)
    }

    # run
    evaluator = contrib.extensions.InstanceSegmentationVOCEvaluator(
        test_iters,
        mask_rcnn,
        device=gpu,
        label_names=class_names,
        show_progress=True,
    )

    reporter = chainer.Reporter()
    reporter.add_observer('main', mask_rcnn)
    with reporter.scope({}):
        result = evaluator.evaluate()
    pprint.pprint(result)


if __name__ == '__main__':
    main()
