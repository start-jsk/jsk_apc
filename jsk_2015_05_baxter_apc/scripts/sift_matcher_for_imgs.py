#!/usr/bin/env python
#
"""
This script is to visualize how match sift features are matched between
an image and camera frame.

Usage
-----

    $ roslaunch roseus_tutorials usb-camera.launch
    $ roslaunch jsk_2015_05_baxter_apc sift_matcher_for_imgs.launch
    $ rosrun image_view image_view image:=/sift_matcher_for_imgs/output

"""
import os

import cv2
import numpy as np

import rospy
import cv_bridge
from sensor_msgs.msg import Image
from posedetection_msgs.srv import Feature0DDetect

from sift_matcher import SiftMatcher, imgsift_client
from common import load_img


class ImageSubscriber(object):
    def __init__(self, image_topic):
        rospy.Subscriber(image_topic, Image, self._cb_img)
        rospy.loginfo('Waiting for: {topic}'.format(topic=image_topic))
        rospy.wait_for_message(image_topic, Image)
        rospy.loginfo('Found: {topic}'.format(topic=image_topic))

    def _cb_img(self, msg):
        """Callback function of Subscribers to listen Image"""
        bridge = cv_bridge.CvBridge()
        self.stamp = msg.header.stamp
        self.img = bridge.imgmsg_to_cv2(msg)


class SiftMatcherOneImg(SiftMatcher):
    """Compare two images.
    Usually camera image (input) with static image (reference)"""
    def __init__(self):
        super(SiftMatcherOneImg, self).__init__()
        self.img_sub = ImageSubscriber('~input')
        self.reference_sub = ImageSubscriber('~input/reference')
        self.pub = rospy.Publisher('~output', Image, queue_size=1)

    def match(self):
        input_stamp, input_img = self.img_sub.stamp, self.img_sub.img
        input_features = self.query_features
        reference_img = self.reference_sub.img
        reference_features = imgsift_client(reference_img)
        matches = self.find_match(input_features.descriptors,
                                  reference_features.descriptors)
        rospy.loginfo('matches: {}'.format(len(matches)))
        # prepare output img
        matched_img = drawMatches(input_img, input_features.positions,
                                  reference_img, reference_features.positions,
                                  matches)
        cv2.putText(matched_img, 'matches: {}'.format(len(matches)),
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        self.publish_img(stamp=input_stamp, img=matched_img)

    def publish_img(self, stamp, img, encoding='bgr8'):
        bridge = cv_bridge.CvBridge()
        img_msg = bridge.cv2_to_imgmsg(img, encoding=encoding)
        img_msg.header.stamp = stamp
        self.pub.publish(img_msg)


def drawMatches(query_img, query_pos, train_img, train_pos, matches):
    """Draw match points for two images"""
    query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
    train_img = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)
    query_pos = np.array(query_pos).reshape((-1, 2))
    train_pos = np.array(train_pos).reshape((-1, 2))
    n_row1, n_col1 = query_img.shape[:2]
    n_row2, n_col2 = train_img.shape[:2]
    # parepare output img
    img_out = np.zeros((max([n_row1,n_row2]), n_col1+n_col2, 3), dtype='uint8')
    img_out[:n_row1, :n_col1, :] = np.dstack(3*[query_img])
    img_out[:n_row2, n_col1:n_col1+n_col2, :] = np.dstack(3*[train_img])
    for mat in matches:
        # draw and connect match points
        x1, y1 = query_pos[mat.queryIdx]
        x2, y2 = train_pos[mat.trainIdx]
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2)+n_col1, int(y2))
        cv2.circle(img_out, pt1, 4, (255, 0, 0), 1)
        cv2.circle(img_out, pt2, 4, (255, 0, 0), 1)
        cv2.line(img_out, pt1, pt2, (255, 0, 0), 1)
    return img_out


def main():
    rospy.init_node('sift_matcher_oneimg')
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        matcher = SiftMatcherOneImg()
        matcher.match()
        rate.sleep()


if __name__ == '__main__':
    main()

