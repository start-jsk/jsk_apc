#!/usr/bin/env python

import os
import os.path as osp
import warnings

import cv2
import fcn
import matplotlib.cm
import numpy as np

import jsk_recognition_utils
import rospkg


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')


def colorize_depth(depth, min_value=None, max_value=None):
    """Colorize depth image with JET colormap."""
    min_value = np.nanmin(depth) if min_value is None else min_value
    max_value = np.nanmax(depth) if max_value is None else max_value
    if np.isinf(min_value) or np.isinf(max_value):
        warnings.warn('Min or max value for depth colorization is inf.')

    colorized = depth.copy()
    nan_mask = np.isnan(colorized)
    colorized[nan_mask] = 0
    colorized = 1. * (colorized - min_value) / (max_value - min_value)
    colorized = matplotlib.cm.jet(colorized)[:, :, :3]
    colorized = (colorized * 255).astype(np.uint8)
    colorized[nan_mask] = (0, 0, 0)
    return colorized


def main():
    dataset_dir = osp.join(PKG_DIR, 'data/datasets/JSK_V1')
    if not osp.exists(dataset_dir):
        print('Please install JSK_V1 dataset to: %s' % dataset_dir)
        quit(1)

    label_files = []
    for stamp_dir in os.listdir(dataset_dir):
        stamp_dir = osp.join(dataset_dir, stamp_dir)
        label_file = osp.join(stamp_dir, 'label.npz')
        if osp.exists(label_file):
            label_files.append(label_file)
        else:
            label_files.append(None)
    print('==> Size of dataset: All: %d, Annotated: %d.' %
          (len(label_files), len(list(filter(None, label_files)))))

    object_names = ['__background__']
    with open(osp.join(PKG_DIR, 'data/names/objects.txt')) as f:
        object_names += [x.strip() for x in f]
    object_names.append('__shelf__')
    object_names.append('__unlabeled__')

    print('==> Press q to quit, and any other keys to go next.')
    for stamp_dir in os.listdir(dataset_dir):
        stamp_dir = osp.join(dataset_dir, stamp_dir)

        img_file = osp.join(stamp_dir, 'image.jpg')
        img = cv2.imread(img_file)

        depth_file = osp.join(stamp_dir, 'depth.npz')
        depth = np.load(depth_file)['arr_0']
        depth_viz = colorize_depth(depth, min_value=0.4, max_value=1.0)

        label_file = osp.join(stamp_dir, 'label.npz')
        if osp.exists(label_file):
            label = np.load(label_file)['arr_0']
            mask_unlabeled = label == -1
            label[mask_unlabeled] = object_names.index('__unlabeled__')
            img_labeled = img.copy()
            img_labeled[mask_unlabeled] = \
                np.random.randint(0, 255, (mask_unlabeled.sum(), 3))
            label_viz = fcn.utils.draw_label(
                label, img_labeled, len(object_names),
                dict(enumerate(object_names)))
        else:
            label_viz = np.zeros_like(img)

        viz = jsk_recognition_utils.get_tile_image([img, depth_viz, label_viz])

        cv2.imshow('view_jsk_v1', viz)
        if cv2.waitKey(0) == ord('q'):
            break

if __name__ == '__main__':
    main()
