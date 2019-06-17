#!/usr/bin/env python

import collections
import glob
import os
import os.path as osp
import shutil

import numpy as np
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split


here = osp.dirname(osp.abspath(__file__))


def main():
    dataset_dir = osp.join(here, 'data/dataset_data/20180730')

    splits_dir = osp.join(here, 'data/dataset_data/20180730_splits')
    os.makedirs(splits_dir)
    shutil.copy(
        osp.join(dataset_dir, 'class_names.txt'),
        osp.join(splits_dir, 'class_names.txt'),
    )

    all_video_dirs = [
        d for d in os.listdir(dataset_dir)
        if osp.isdir(osp.join(dataset_dir, d))
    ]

    count_min_train = 6
    count_max_train = 11
    count_min_test = 3
    count_max_test = 7

    while True:
        train_video_dirs, test_video_dirs = train_test_split(
            all_video_dirs, train_size=0.66
        )

        counter_all = collections.Counter()
        counters = []
        for video_dirs in [train_video_dirs, test_video_dirs]:
            counter = collections.Counter()
            for video_dir in sorted(video_dirs):
                video_dir = osp.join(dataset_dir, video_dir)
                if not osp.isdir(video_dir):
                    continue

                labels_all = set()
                npz_files = sorted(glob.glob(osp.join(video_dir, '*.npz')))
                for npz_file in npz_files:
                    data = np.load(npz_file)
                    labels = data['labels']
                    labels_all.update(labels)

                counter.update(labels_all)

                counter_all.update(labels_all)

            counters.append(counter)

        ok = True
        for split, counter in zip(['train', 'test'], counters):
            if split == 'train':
                count_min = count_min_train
                count_max = count_max_train
            else:
                count_min = count_min_test
                count_max = count_max_test
            print(split, min(counter.values()), max(counter.values()))
            if not (min(counter.values()) >= count_min and
                    max(counter.values()) <= count_max):
                ok = False

        # for label, count in sorted(counter_all.items()):
        #     print('%02d' % label, count)

        if ok:
            for video_dir in train_video_dirs:
                shutil.copytree(
                    osp.join(dataset_dir, video_dir),
                    osp.join(splits_dir, 'train', video_dir),
                )
            for video_dir in test_video_dirs:
                shutil.copytree(
                    osp.join(dataset_dir, video_dir),
                    osp.join(splits_dir, 'test', video_dir),
                )
            print('Saved to: {}'.format(splits_dir))

            break


if __name__ == '__main__':
    main()
