#!/usr/bin/env python

import os
import os.path as osp
import shutil

import numpy as np
import skimage.io
import yaml

import jsk_data


ROS_HOME = osp.expanduser('~/.ros')


class DatasetCollectedOnShelfMultiView(object):

    def __init__(self):
        self.ids = []
        self.root = osp.join(
            ROS_HOME, 'jsk_arc2017_common/dataset_jsk_v3_20160614')

        # download from google drive
        try:
            os.makedirs(osp.dirname(self.root))
        except OSError:
            pass
        jsk_data.download_data(
            pkg_name='jsk_arc2017_common',
            url='https://drive.google.com/uc?id=0B1waEZT19wijb3dMZXhhcnp3Szg',
            path=self.root + '.tgz',
            md5='0d33a08aa5f64e0d880a7b8ca34c6ab7',
            extract=True,
        )

        for id_ in sorted(os.listdir(self.root)):
            self.ids.append(id_)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        data_dir = osp.join(self.root, id_)

        frame_idx = int(
            open(osp.join(data_dir, 'view_frame.txt')).read().strip())
        img = skimage.io.imread(osp.join(data_dir, 'image.jpg'))
        depth = np.load(osp.join(data_dir, 'depth.npz'))['arr_0']
        camera_info = yaml.load(
            open(osp.join(data_dir,
                          'camera_info_right_hand_left_camera.yaml')))
        tf_camera_from_base = yaml.load(
            open(osp.join(data_dir, 'tf_camera_rgb_from_base.yaml')))

        return frame_idx, img, depth, camera_info, tf_camera_from_base


def main():
    dataset = DatasetCollectedOnShelfMultiView()

    out_dir = osp.join(
        ROS_HOME, 'jsk_arc2017_common/dataset_jsk_v3_20160614_scenes')
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    scene_idx = 0
    for i in range(len(dataset)):
        frame_idx, img, depth = dataset[i][:3]
        if frame_idx == 1:
            scene_idx += 1

        scene_dir = osp.join(out_dir, 'scene-%04d' % scene_idx)
        if not osp.exists(scene_dir):
            os.makedirs(scene_dir)

        frame_dir = osp.join(dataset.root, dataset.ids[i])
        shutil.copytree(frame_dir, osp.join(scene_dir, dataset.ids[i]))
        print('%s -> %s' % (frame_dir, osp.join(scene_dir, dataset.ids[i])))

        # cv2.imshow('create_scenes_dataset_v3', img[:, :, ::-1])
        # k = cv2.waitKey(0)
        # if k == ord('q'):
        #     break


if __name__ == '__main__':
    main()
