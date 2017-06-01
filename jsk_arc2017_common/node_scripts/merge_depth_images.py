#!/usr/bin/env python

import collections
import sys

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image


class MergeDepthImages(ConnectionBasedTransport):

    def __init__(self):
        super(MergeDepthImages, self).__init__()
        self.pub = self.advertise('~output', Image, queue_size=1)

    def subscribe(self):
        input_topics = rospy.get_param('~input_topics')
        if not (isinstance(input_topics, collections.Sequence) and
                len(input_topics) > 1):
            rospy.logfatal('~input_topics must be list and > 1')
            sys.exit(1)

        self.subs = []
        for topic in rospy.get_param('~input_topics'):
            self.subs.append(
                message_filters.Subscriber(topic, Image, queue_size=1,
                                           buff_size=2**24))
        queue_size = rospy.get_param('~queue_size', 10)
        slop = rospy.get_param('~slop', 0.1)
        sync = message_filters.ApproximateTimeSynchronizer(
            fs=self.subs, queue_size=queue_size, slop=slop)
        sync.registerCallback(self._cb)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _cb(self, *msgs):
        encoding = msgs[0].encoding
        height, width = msgs[0].height, msgs[0].width
        if not all(encoding == msg.encoding for msg in msgs):
            # can be 32FC1 or 16UC1
            rospy.logerr('Input msgs must have same encodings.')
            return
        if not all((height == msg.height and width == msg.width)
                   for msg in msgs):
            rospy.logerr('Input msgs must have same image sizes.')
            return

        bridge = cv_bridge.CvBridge()

        dtype, n_channels = bridge.encoding_to_dtype_with_channels(encoding)
        assert n_channels == 1
        depth_merged = np.zeros((height, width), dtype=dtype)
        n_fusions = np.zeros((height, width), dtype=np.uint32)

        bridge = cv_bridge.CvBridge()
        for msg in msgs:
            depth = bridge.imgmsg_to_cv2(msg)
            if depth.dtype == np.uint16:
                nonnan_mask = depth != 0
            elif depth.dtype == np.float32:
                nonnan_mask = ~np.isnan(depth)
            else:
                raise ValueError
            depth_merged[nonnan_mask] += depth[nonnan_mask]
            n_fusions[nonnan_mask] += 1

        mask_fusioned = n_fusions > 0
        depth_merged[mask_fusioned] = \
            depth_merged[mask_fusioned] / n_fusions[mask_fusioned]
        if depth.dtype == np.uint16:
            depth_merged[n_fusions == 0] = 0
        elif depth.dtype == np.float32:
            depth_merged[n_fusions == 0] = np.nan
        else:
            raise ValueError

        out_msg = bridge.cv2_to_imgmsg(depth_merged)
        out_msg.header = msgs[0].header
        self.pub.publish(out_msg)


if __name__ == '__main__':
    rospy.init_node('merge_depth_images')
    MergeDepthImages()
    rospy.spin()
