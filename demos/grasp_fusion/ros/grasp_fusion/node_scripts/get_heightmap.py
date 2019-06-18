#!/usr/bin/env python

import cv_bridge
import message_filters
import numpy as np
import rospy
import tf
# For debug
# import grasp_fusion_lib

from jsk_topic_tools import ConnectionBasedTransport
from jsk_topic_tools.log_utils import logerr_throttle
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from grasp_fusion_lib.contrib import grasp_fusion


class GetHeightmap(ConnectionBasedTransport):

    def __init__(self):
        super(GetHeightmap, self).__init__()

        self.heightmap_frame = rospy.get_param('~heightmap_frame')
        # Size[m] of each height map pixel
        self.voxel_size = rospy.get_param('~voxel_size')

        self.listener = tf.TransformListener()
        self.tft = tf.TransformerROS()
        self.br = cv_bridge.CvBridge()

        # ROS publishers
        self.pub_rgb = self.advertise('~output/rgb', Image, queue_size=1)
        self.pub_depth = self.advertise('~output/depth', Image, queue_size=1)
        self.pub_label = self.advertise('~output/label', Image, queue_size=1)

        self._bg_label = rospy.get_param('~bg_label', 0)

    def subscribe(self):
        self.sub_rgb = message_filters.Subscriber(
            '~input/rgb', Image, queue_size=1, buff_size=2**24
        )
        self.sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1, buff_size=2**24
        )
        self.sub_info = message_filters.Subscriber(
            '~input/camera_info', CameraInfo
        )
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info],
            queue_size=100,
            slop=0.1,
        )
        sync.registerCallback(self.callback, 'rgb')

        self.sub_label = message_filters.Subscriber(
            '~input/label', Image, queue_size=1, buff_size=2**24
        )
        sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_label, self.sub_depth, self.sub_info],
            queue_size=100,
            slop=0.1,
        )
        sync.registerCallback(self.callback, 'label')

    def unsubscribe(self):
        self.sub_rgb.unregister()
        self.sub_depth.unregister()
        self.sub_info.unregister()
        self.sub_label.unregister()

    def callback(self, img_input, depth_input, cam_info, mode):
        assert mode in ['rgb', 'label']

        # From tf, generate camera pose w.r.t heightmap_frame
        try:
            trans, rot \
                = self.listener.lookupTransform(self.heightmap_frame,
                                                img_input.header.frame_id,
                                                rospy.Time(0))
        except Exception as e:
            logerr_throttle(10, e)
            return

        cam_pose = self.tft.fromTranslationRotation(trans, rot)
        # Generate other data
        cam_intrinsics = np.array(cam_info.K).reshape(3, 3)

        if mode == 'rgb':
            color_img = self.br.imgmsg_to_cv2(
                img_input, desired_encoding='rgb8'
            )
            color_img = color_img.astype(float) / 255  # Convert to range [0,1]
            label_img = np.zeros(
                (color_img.shape[0], color_img.shape[1]), dtype=np.int32
            )
        else:
            label_img = self.br.imgmsg_to_cv2(
                img_input, desired_encoding='passthrough'
            )
            # this should be filled by 1 for bg subtraction in get_heightmap
            color_img = np.ones(
                (label_img.shape[0], label_img.shape[1], 3), dtype=float
            )

        depth_img = self.br.imgmsg_to_cv2(
            depth_input, desired_encoding='passthrough'
        )
        # Replace nan element to zero
        depth_img = np.where(np.isnan(depth_img), 0, depth_img)
        if depth_input.encoding == '16UC1':
            depth_img = depth_img.astype(float) / 1000.0  # Convert mm to m
        elif depth_input.encoding != '32FC1':
            enc = depth_input.encoding
            logerr_throttle(10, 'Unsupported depth encoding: %s' % enc)
            return

        # Generate heightmap w.r.t heightmap_frame
        heightmap_color, heightmap, missing_heightmap, heightmap_label \
            = grasp_fusion.utils.get_heightmap(
                color_img=color_img,
                depth_img=depth_img,
                bg_color_img=np.zeros_like(color_img),
                bg_depth_img=np.zeros_like(depth_img),
                cam_intrinsics=cam_intrinsics,
                cam_pose=cam_pose,
                grid_origin=np.array([0, 0, 0]),
                grid_rot=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                suction_img=label_img,
                voxel_size=self.voxel_size,
                suction_cval=self._bg_label,
            )

        color_data, depth_data \
            = grasp_fusion.utils.heightmap_postprocess(
                heightmap_color,
                heightmap,
                missing_heightmap,
            )
        # it is scaled in postprocess
        depth_data = (depth_data / 10000.).astype(np.float32)

        heightmap_label = heightmap_label.reshape(
            heightmap.shape[0], heightmap.shape[1],
        )
        # Consider pixels whose height is 0 as background
        heightmap_label[heightmap == 0] = self._bg_label
        label_data = np.full((224, 320), self._bg_label, dtype=label_img.dtype)
        label_data[12:212, 10:310] = heightmap_label

        # For debug
        # depth = grasp_fusion_lib.image.colorize_depth(depth_data,
        #                                   min_value=0, max_value=1.5)
        # viz = grasp_fusion_lib.image.tile([color_data, depth], (1, 2))
        # grasp_fusion_lib.io.imshow(viz)
        # grasp_fusion_lib.io.waitkey()

        if mode == 'rgb':
            rgb_output = self.br.cv2_to_imgmsg(color_data, encoding='rgb8')
            rgb_output.header = img_input.header
            self.pub_rgb.publish(rgb_output)
        else:
            assert mode == 'label'
            label_output = self.br.cv2_to_imgmsg(label_data)
            label_output.header = img_input.header
            self.pub_label.publish(label_output)

        depth_output = self.br.cv2_to_imgmsg(
            depth_data, encoding='passthrough'
        )
        depth_output.header = img_input.header
        self.pub_depth.publish(depth_output)


if __name__ == '__main__':
    rospy.init_node('get_heightmap')
    get_heightmap = GetHeightmap()
    rospy.spin()
