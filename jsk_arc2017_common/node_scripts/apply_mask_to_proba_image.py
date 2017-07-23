#!/usr/bin/env python

import cv_bridge
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image


class ApplyMaskToProbaImage():

    def __init__(self):
        self.pub_img = None
        self.pub = rospy.Publisher('~output', Image, queue_size=1)
        self.subscribe()

    def subscribe(self):
        use_async = rospy.get_param('~approximate_sync', False)
        queue_size = rospy.get_param('~queue_size', 10)
        proba_sub = message_filters.Subscriber('~input_proba', Image)
        mask_sub = message_filters.Subscriber('~input_mask', Image)
        if use_async:
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                [proba_sub, mask_sub], queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                [proba_sub, mask_sub], queue_size=queue_size)
        sync.registerCallback(self.callback)

    def callback(self, proba_msg, mask_msg):
        br = cv_bridge.CvBridge()
        proba_img = br.imgmsg_to_cv2(proba_msg).copy()
        mask = br.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        if mask.ndim > 2:
            mask = np.squeeze(mask, axis=2)
        proba_img[:, :, 0][mask == 0] = 1
        proba_img[:, :, 1:][mask == 0] = 0
        proba_msg_masked = br.cv2_to_imgmsg(proba_img.astype(np.float32))
        proba_msg_masked.header = proba_msg.header
        self.pub.publish(proba_msg_masked)


if __name__ == '__main__':
    rospy.init_node('apply_mask_to_proba_image')
    app = ApplyMaskToProbaImage()
    rospy.spin()
