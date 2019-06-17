#!/usr/bin/env python

import chainer
import cv2

import chainer_mask_rcnn
import instance_occlsegm_lib

import contrib


def visualize_func(dataset, index):
    img, bboxes, labels, masks = dataset[index]
    captions = [dataset.class_names[l] for l in labels]
    img_gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                            cv2.COLOR_GRAY2RGB)
    viz_ins = chainer_mask_rcnn.utils.draw_instance_bboxes(
        img_gray, bboxes=bboxes, labels=labels,
        n_class=len(dataset.class_names),
        masks=masks, captions=captions)
    return instance_occlsegm_lib.image.tile([img, viz_ins])


dataset = contrib.datasets.ARC2017RealInstancesDataset(split='train')
class_names = dataset.class_names
dataset = chainer.datasets.ConcatenatedDataset(
    dataset,
    contrib.datasets.ARC2017RealInstancesDataset(split='test'))
dataset.class_names = class_names
instance_occlsegm_lib.datasets.view_dataset(dataset, visualize_func)
