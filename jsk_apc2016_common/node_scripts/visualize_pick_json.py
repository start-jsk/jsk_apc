#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import matplotlib.pyplot as plt

import cv_bridge
import rospy
from sensor_msgs.msg import Image

import jsk_apc2016_common


def visualize_cb(event):
    if pub.get_num_connections() > 0:
        imgmsg.header.stamp = rospy.Time.now()
        pub.publish(imgmsg)
    if display:
        global displayed_img
        img_rgb = img[:, :, ::-1]
        plt.axis('off')
        plt.tight_layout()
        if displayed_img and displayed_img._imcache is None:
            plt.close()
        if displayed_img is None:
            displayed_img = plt.imshow(img_rgb)
        else:
            displayed_img.set_data(img_rgb)
        plt.pause(0.01)


if __name__ == '__main__':
    rospy.init_node('visualize_pick_json')
    pub = rospy.Publisher('~output', Image, queue_size=10)

    parser = argparse.ArgumentParser()
    parser.add_argument('json',
                        help='JSON file with bin_contents and work_order')
    parser.add_argument('-d', '--display', action='store_true',
                        help='Display with a window')
    args = parser.parse_args(rospy.myargv()[1:])
    json = args.json
    display = args.display
    displayed_img = None

    img = jsk_apc2016_common.visualize_pick_json(json)

    br = cv_bridge.CvBridge()
    imgmsg = br.cv2_to_imgmsg(img, encoding='bgr8')

    timer = rospy.Timer(rospy.Duration(0.1), visualize_cb)
    rospy.spin()
