#!/usr/bin/env python

import os
import os.path as osp

import numpy as np
import skimage.io
import yaml

import cv_bridge
import dynamic_reconfigure.server
from geometry_msgs.msg import TransformStamped
import genpy.message
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import tf

from jsk_arc2017_common.cfg import PublishDatasetV3Config


class DatasetCollectedOnShelfMultiViewScenes(object):

    def __init__(self):
        self.scene_ids = []
        self.root = '/data/projects/arc2017/datasets/JSKV3_scenes'
        for scene_id in sorted(os.listdir(self.root)):
            self.scene_ids.append(scene_id)

    def __len__(self):
        return len(self.scene_ids)

    def get_frame(self, scene_idx, frame_idx=0):
        assert 0 <= scene_idx < len(self.scene_ids)
        assert 0 <= frame_idx < 9
        scene_id = self.scene_ids[scene_idx]
        scene_dir = osp.join(self.root, scene_id)
        frame_dirs = sorted(os.listdir(scene_dir))
        frame_dir = osp.join(scene_dir, frame_dirs[frame_idx])

        frame_id = int(
            open(osp.join(frame_dir, 'view_frame.txt')).read().strip())
        assert frame_id == frame_idx + 1
        img = skimage.io.imread(osp.join(frame_dir, 'image.jpg'))
        depth = np.load(osp.join(frame_dir, 'depth.npz'))['arr_0']
        camera_info = yaml.load(
            open(osp.join(frame_dir,
                          'camera_info_right_hand_camera_left.yaml')))
        tf_camera_from_base = yaml.load(
            open(osp.join(frame_dir, 'tf_camera_rgb_from_base.yaml')))

        return frame_id, img, depth, camera_info, tf_camera_from_base


class PublishDatasetV3(object):

    def __init__(self):
        self._dataset = DatasetCollectedOnShelfMultiViewScenes()

        self._config_srv = dynamic_reconfigure.server.Server(
            PublishDatasetV3Config, self._config_cb)

        self.pub_rgb = rospy.Publisher(
            '~output/rgb/image_rect_color', Image, queue_size=1)
        self.pub_rgb_cam_info = rospy.Publisher(
            '~output/rgb/camera_info', CameraInfo, queue_size=1)
        self.pub_depth = rospy.Publisher(
            '~output/depth_registered/image_rect', Image, queue_size=1)
        self.pub_depth_cam_info = rospy.Publisher(
            '~output/depth_registered/camera_info', CameraInfo, queue_size=1)

        self.tf_broadcaster = tf.broadcaster.TransformBroadcaster()
        self._timer = rospy.Timer(rospy.Duration(1. / 30), self._timer_cb)

    def _config_cb(self, config, level):
        self._scene_idx = config.scene_idx
        self._frame_idx = config.frame_idx
        return config

    def _timer_cb(self, event):
        img, depth, cam_info, tf = self._dataset.get_frame(
            self._scene_idx, self._frame_idx)[1:5]

        cam_info_msg = CameraInfo()
        genpy.message.fill_message_args(cam_info_msg, cam_info)
        cam_info_msg.header.stamp = event.current_real

        tf_msg = TransformStamped()
        genpy.message.fill_message_args(tf_msg, tf)
        tf_msg.header.stamp = event.current_real

        bridge = cv_bridge.CvBridge()

        imgmsg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
        imgmsg.header.frame_id = cam_info_msg.header.frame_id
        imgmsg.header.stamp = event.current_real

        depth_msg = bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header.frame_id = cam_info_msg.header.frame_id
        depth_msg.header.stamp = event.current_real

        self.tf_broadcaster.sendTransformMessage(tf_msg)
        self.pub_rgb.publish(imgmsg)
        self.pub_rgb_cam_info.publish(cam_info_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_depth_cam_info.publish(cam_info_msg)


if __name__ == '__main__':
    rospy.init_node('publish_dataset_v3')
    app = PublishDatasetV3()
    rospy.spin()
