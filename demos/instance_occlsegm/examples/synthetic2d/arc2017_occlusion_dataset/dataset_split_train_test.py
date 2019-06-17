#!/usr/bin/env python

import collections
import glob
import os
import os.path as osp
import shutil

import numpy as np


here = osp.dirname(osp.abspath(__file__))


def main():
    dataset_dir = osp.join(here, 'dataset_data/20180204')
    splits_dir = osp.join(here, 'dataset_data/20180204_splits')

    if osp.exists(splits_dir):
        print('splits_dir already exists: %s' % splits_dir)
        return
    os.makedirs(splits_dir)
    shutil.copy(
        osp.join(dataset_dir, 'class_names.txt'),
        osp.join(splits_dir, 'class_names.txt'))

    object_freq_by_video = {class_id: 0 for class_id in range(1, 41)}
    object_freq_by_frame = {class_id: 0 for class_id in range(1, 41)}
    videos = []
    for video_id in sorted(os.listdir(dataset_dir)):
        video_dir = osp.join(dataset_dir, video_id)
        if not osp.isdir(video_dir):
            continue

        npz_files = sorted(glob.glob(osp.join(video_dir, '*.npz')))

        n_frames = 0
        class_ids_in_video = set()
        for npz_file in npz_files:
            lbl_cls = np.load(npz_file)['lbl_cls']

            class_ids = np.unique(lbl_cls)
            keep = ~np.isin(class_ids, [-1, 0])
            class_ids = class_ids[keep]

            for class_id in class_ids:
                object_freq_by_frame[class_id] += 1
                class_ids_in_video.add(class_id)

            n_frames += 1

        for class_id in class_ids_in_video:
            object_freq_by_video[class_id] += 1

        videos.append(dict(
            id=video_id,
            dir=video_dir,
            class_ids=list(sorted(class_ids_in_video)),
            n_objects=len(class_ids_in_video),
            n_frames=n_frames,
        ))

    print('# of videos: %d' % len(videos))

    # RANSAC to split Train/Test
    ratio_train = 0.66
    n_train = int(ratio_train * len(videos))
    while True:
        p = np.random.permutation(len(videos))
        indices_train = p[:n_train]
        indices_test = p[n_train:]
        videos_train = [videos[i] for i in indices_train]
        videos_test = [videos[i] for i in indices_test]

        class_ids = []
        for video in videos_train:
            class_ids.extend(video['class_ids'])
        count = collections.Counter(class_ids)
        count_values_unique = set(count.values())
        if not count_values_unique.issubset({3, 4, 5}):
            continue

        mean_count_ideal = 7 * ratio_train
        mean_count = 1. * sum(count.values()) / len(count)
        if abs(mean_count - mean_count_ideal) > 0.1:
            continue

        break

    print('Mean Count (Ideal): %f' % mean_count_ideal)
    print('Mean Count: %f' % mean_count)
    # print(count_values_unique)
    # print(count.values())

    print('Videos Train: %s' % sorted([v['id'] for v in videos_train]))
    print('Videos Test: %s' % sorted([v['id'] for v in videos_test]))

    split_dir = osp.join(splits_dir, 'train')
    os.makedirs(split_dir)
    for video in videos_train:
        shutil.copytree(
            video['dir'], osp.join(split_dir, osp.basename(video['dir'])))
    split_dir = osp.join(splits_dir, 'test')
    os.makedirs(split_dir)
    for video in videos_test:
        shutil.copytree(
            video['dir'], osp.join(split_dir, osp.basename(video['dir'])))

    print('Splitted dataset: %s' % splits_dir)


if __name__ == '__main__':
    main()
