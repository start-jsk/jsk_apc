#!/usr/bin/env python

import argparse
import glob
import os.path as osp

import numpy as np

import chainer_mask_rcnn as mrcnn
import instance_occlsegm_lib


here = osp.dirname(osp.abspath(__file__))


class ARC2017OcclusionVideoDataset(object):

    def __init__(self, video_dir, class_names):
        self._npz_files = sorted(glob.glob(osp.join(video_dir, '*.npz')))
        self.class_names = class_names

    def __len__(self):
        return len(self._npz_files)

    def __getitem__(self, i):
        npz_file = self._npz_files[i]
        data = np.load(npz_file)
        return data['img'], data['bboxes'], data['labels'], data['masks']


def main():
    default_video_dir = osp.join(here, 'dataset_data/20180204/0000')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video_dir', default=default_video_dir,
                        help='video_dir', nargs='?')
    args = parser.parse_args()

    print('Viewing video: %s' % args.video_dir)

    class_names_file = osp.join(default_video_dir, '../class_names.txt')
    class_names = [n.strip() for n in open(class_names_file)]

    data = ARC2017OcclusionVideoDataset(args.video_dir, class_names)

    def visualize_func(dataset, index):
        print('Index: %08d' % index)

        img, bboxes, labels, masks = dataset[index]

        captions = ['%d: %s' % (l, dataset.class_names[l]) for l in labels]
        viz1 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=41, masks=masks == 1,
            captions=captions)
        viz2 = mrcnn.utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=41, masks=masks == 2,
            captions=captions)

        return np.hstack([img, viz1, viz2])

    instance_occlsegm_lib.datasets.view_dataset(data, visualize_func)


if __name__ == '__main__':
    main()
