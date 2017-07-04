#!/usr/bin/env python

import os
import os.path as osp
import shutil

# import cv2
import numpy as np
import skimage.io
import yaml

import cv_bridge
from geometry_msgs.msg import TransformStamped
import genpy.message
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import tf


class DatasetCollectedOnShelfMultiView(object):

    def __init__(self):
        self.ids = []
        self.root = '/data/projects/arc2017/datasets/JSKV3'
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
                          'camera_info_right_hand_camera_left.yaml')))
        tf_camera_from_base = yaml.load(
            open(osp.join(data_dir, 'tf_camera_rgb_from_base.yaml')))

        return frame_idx, img, depth, camera_info, tf_camera_from_base


def main():
    dataset = DatasetCollectedOnShelfMultiView()

    out_dir = '/data/projects/arc2017/datasets/JSKV3_scenes'
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    scene_idx = 0
    for i in xrange(len(dataset)):
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
