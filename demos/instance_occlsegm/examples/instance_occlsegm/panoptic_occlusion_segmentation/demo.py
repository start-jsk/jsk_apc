#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
import imgviz
import yaml

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import instance_occlsegm


class Images(object):

    def __init__(self, model, dataset):
        self._model = model
        self._dataset = dataset
        self._fg_class_names = dataset.class_names

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        example = self._dataset[index]

        img = example[0]
        img = imgviz.centerize(img, (480, 640), cval=0)

        img_chw = img.transpose(2, 0, 1)
        bboxes, masks, labels, scores, lbls_vis, lbls_occ = \
            self._model.predict([img_chw])

        bbox = bboxes[0]
        mask = masks[0]
        label = labels[0]
        # score = scores[0]
        # lbl_vis = lbls_vis[0]

        n_instance = len(bbox)
        vizs = [img]
        for i in range(n_instance):
            viz = instance_occlsegm_lib.image.label2rgb(
                lbl=mask[i],
                img=img,
            )
            viz = imgviz.instances2rgb(
                image=viz,
                labels=[label[i] + 1],
                bboxes=[bbox[i]],
                captions=[self._fg_class_names[label[i]]],
            )
            vizs.append(viz)
        viz = imgviz.tile(vizs, border=(255,) * 3, shape=(4, 4))
        viz = imgviz.resize(viz, height=800)

        return viz


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('log_dir')
    args = parser.parse_args()

    pretrained_model = osp.join(args.log_dir, 'snapshot_model.npz')
    yaml_file = osp.join(args.log_dir, 'params.yaml')

    with open(yaml_file) as f:
        params = yaml.safe_load(f)

    n_layers = int(params['model'].lstrip('resnet'))
    model = instance_occlsegm.models.MaskRCNNPanopticResNet(
        n_layers=n_layers,
        n_fg_class=40,
        pretrained_model=pretrained_model,
        min_size=params['min_size'],
        max_size=params['max_size'],
        anchor_scales=params['anchor_scales'],
        rpn_dim=params['rpn_dim'],
    )
    model.nms_thresh = 0.3
    model.score_thresh = 0.7

    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()

    dataset = instance_occlsegm.datasets.PanopticOcclusionSegmentationDataset(
        'test'
    )

    images = Images(model=model, dataset=dataset)

    imgviz.io.pyglet_imshow(images)
    imgviz.io.pyglet_run()


if __name__ == '__main__':
    main()
