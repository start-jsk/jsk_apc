#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import cv2

import rospy
import cv_bridge
from sensor_msgs.msg import Image
from jsk_2015_05_baxter_apc.srv import StringEmpty


class ImagePublishServer(object):
    """Publish requested image file"""

    def __init__(self, encoding='bgr8'):
        rospy.Service('image_publish_server', StringEmpty, self.callback)
        self.pub_of_img = rospy.Publisher('~output', Image, queue_size=1)
        self.filename = None
        self.encoding = encoding

    def callback(self, req):
        self.filename = req.string
        self.spin_once()
        return []

    def spin_once(self):
        filename = self.filename
        if filename is None:
            return
        img = cv2.imread(filename)
        if img is None:
            rospy.logwarn('file not found')
            self.filename = None
            return
        bridge = cv_bridge.CvBridge()
        imgmsg = bridge.cv2_to_imgmsg(img, encoding=self.encoding)
        imgmsg.header.stamp = rospy.Time.now()
        self.pub_of_img.publish(imgmsg)

    def spin(self):
        rate = rospy.Rate(rospy.get_param("rate", 1))
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('image_publish_server')
    impub_server = ImagePublishServer()
    impub_server.spin()

