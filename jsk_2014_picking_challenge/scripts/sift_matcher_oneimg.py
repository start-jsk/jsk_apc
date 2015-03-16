#!/usr/bin/env python
#
"""
This script is to visualize how match sift features are matched between
an image and camera frame.

Usage
-----

    $ roslaunch jsk_2014_picking_challenge \
        sift_matcher_oneimg_usbcamera.launch
    $ rosrun jsk_2014_picking_challenge sift_matcher_oneimg.py \
        _imgfile:=`rospack find jsk_2014_picking_challenge`/data/paste.png \
        _maskfile:= \
            `rospack find jsk_2014_picking_challenge`/data/paste_mask.png
    $ rosrun image_view image_view image:=/sift_matcher_oneimg/output

"""
import os

import cv2
import numpy as np

import rospy
import cv_bridge
from sensor_msgs.msg import Image

from sift_matcher import SiftMatcher, imgsift_client
from matcher_common import load_img


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
    def __init__(self, rawfile, maskfile):
        super(SiftMatcherOneImg, self).__init__()
        self.img_sub = ImageSubscriber('~input')
        self.pub = rospy.Publisher('~output', Image, queue_size=1)
        train_img = load_img(rawfile)
        mask_img = load_img(maskfile)
        self.train_img = cv2.add(mask_img, train_img)
        self.train_features = imgsift_client(train_img)

    def match(self):
        stamp, query_img = self.img_sub.stamp, self.img_sub.img
        query_features = self.query_features
        train_img, train_features = self.train_img, self.train_features
        matches = self.find_match(query_features.descriptors,
                                  train_features.descriptors)
        rospy.loginfo('matches: {}'.format(len(matches)))
        # prepare output img
        matched_img = drawMatches(query_img, query_features.positions,
                                  train_img, train_features.positions,
                                  matches)
        cv2.putText(matched_img, 'matches: {}'.format(len(matches)),
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        self.publish_img(stamp=stamp, img=matched_img)

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
    # get params
    rawfile = rospy.get_param('~rawfile', 'image.png')
    base, ext = os.path.splitext(rawfile)
    maskfile = rospy.get_param('~maskfile',
        '{base}_mask{ext}'.format(base=base, ext=ext))
    rospy.loginfo('rawfile: {raw}'.format(raw=rawfile))
    rospy.loginfo('maskfile: {mask}'.format(mask=maskfile))

    while not rospy.is_shutdown():
        matcher = SiftMatcherOneImg(rawfile, maskfile)
        matcher.match()
        rate.sleep()


if __name__ == '__main__':
    main()

