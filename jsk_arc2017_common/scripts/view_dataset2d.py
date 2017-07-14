#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import warnings

import cv2
import dateutil.parser
import matplotlib.cm
import numpy as np
import skimage.io

import jsk_recognition_utils
import rospkg
import yaml


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


def get_text_color(color):
    if color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114 > 170:
        return (0, 0, 0)
    return (255, 255, 255)


def label2rgb(lbl, img=None, label_names=None, alpha=0.3):
    import fcn
    import scipy
    if label_names is None:
        n_labels = lbl.max() + 1  # +1 for bg_label 0
    else:
        n_labels = len(label_names)
    cmap = fcn.utils.labelcolormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]

    if img is not None:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    if label_names is None:
        return lbl_viz

    np.random.seed(1234)
    labels = np.unique(lbl)
    labels = labels[labels != 0]
    for label in labels:
        mask = lbl == label
        mask = (mask * 255).astype(np.uint8)
        y, x = scipy.ndimage.center_of_mass(mask)
        y, x = map(int, [y, x])

        if lbl[y, x] != label:
            Y, X = np.where(mask)
            point_index = np.random.randint(0, len(Y))
            y, x = Y[point_index], X[point_index]

        text = label_names[label]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness)

        color = get_text_color(lbl_viz[y, x])
        cv2.putText(lbl_viz, text,
                    (x - text_size[0] // 2, y),
                    font_face, font_scale, color, thickness)
    return lbl_viz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='Dataset dir path')
    parser.add_argument('-s', '--start',
                        help='Start timestamp (ex. 2017-06-10T10:00:23)')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    start = args.start

    if not osp.exists(dataset_dir):
        print('Please install dataset to: %s' % dataset_dir)
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

    PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')
    with open(osp.join(PKG_DIR, 'config/label_names.yaml')) as f:
        object_names = yaml.load(f)
    object_names.append('__unlabeled__')

    print('==> Press keys: [q] to quit, [n] to go next, [p] to go previous')
    stamp_dirs = list(sorted(os.listdir(dataset_dir)))
    i = 0
    while True:
        stamp = datetime.datetime.fromtimestamp(int(stamp_dirs[i]) / 1e9)
        if start and stamp < dateutil.parser.parse(start):
            i += 1
            continue
        start = None

        stamp_dir = osp.join(dataset_dir, stamp_dirs[i])
        print('%s: %s' % (stamp.isoformat(), stamp_dir))

        img_file = osp.join(stamp_dir, 'image.jpg')
        img = skimage.io.imread(img_file)

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
            label_viz = label2rgb(
                lbl=label, img=img_labeled,
                label_names=dict(enumerate(object_names)))
        else:
            label_viz = np.zeros_like(img)

        viz = jsk_recognition_utils.get_tile_image([img, label_viz, depth_viz])

        cv2.imshow('view_jsk_v1', viz[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('n'):
            if i == len(stamp_dirs) - 1:
                print('Reached the end edge of the dataset')
                continue
            i += 1
        elif key == ord('p'):
            if i == 0:
                print('Reached the start edge of the dataset')
                continue
            i -= 1
        else:
            continue


if __name__ == '__main__':
    main()
