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
import itertools

import cv2
import numpy as np

import rospy
import cv_bridge
from sensor_msgs.msg import Image
from posedetection_msgs.msg import ImageFeature0D
from posedetection_msgs.srv import Feature0DDetect


class SiftMatcherOneImg(object):
    def __init__(self, rawfile, maskfile):
        # Subscribers
        sub_imgfeature = rospy.Subscriber('/ImageFeature0D', ImageFeature0D,
                                          self._cb_imgfeature)
        sub_img = rospy.Subscriber('/image', Image, self._cb_img)
        # Publishers
        self.pub = rospy.Publisher('~output', Image, queue_size=1)
        # train img
        train_img = cv2.imread(rawfile)
        mask_img = cv2.imread(maskfile)
        self.train_img = cv2.add(mask_img, train_img)
        self.train_features = self.imgsift_client(train_img)
        # query img
        self.query_img = None
        self.query_features = None
        rospy.wait_for_message('/image', Image)
        rospy.wait_for_message('/ImageFeature0D', ImageFeature0D)

    def _cb_imgfeature(self, msg):
        """Callback function of Subscribers to listen ImageFeature0D"""
        self.query_features = msg.features

    def _cb_img(self, msg):
        """Callback function of Subscribers to listen Image"""
        bridge = cv_bridge.CvBridge()
        self.query_img = bridge.imgmsg_to_cv2(msg)

    @staticmethod
    def imgsift_client(img):
        """Request to imagesift with Image as service client"""
        client = rospy.ServiceProxy('/Feature0DDetect', Feature0DDetect)
        client.wait_for_service(10)
        bridge = cv_bridge.CvBridge()
        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        resp = client(img_msg)
        return resp.features

    @staticmethod
    def find_match(query_des, train_des):
        """Find match points of query and train images"""
        # parepare to match keypoints
        query_des = np.array(query_des).reshape((-1, 128))
        query_des = (query_des * 255).astype('uint8')
        train_des = np.array(train_des).reshape((-1, 128))
        train_des = (train_des * 255).astype('uint8')
        # find good match points
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(query_des, train_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75*n.distance]
        return good_matches

    @staticmethod
    def drawMatches(query_img, query_pos, train_img, train_pos, matches):
        """Draw match points for two images"""
        query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
        train_img = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)
        query_pos = np.array(query_pos).reshape((-1, 2))
        train_pos = np.array(train_pos).reshape((-1, 2))
        n_row1, n_col1 = query_img.shape[:2]
        n_row2, n_col2 = train_img.shape[:2]
        # parepare output img
        img_out = np.zeros((max([n_row1,n_row2]), n_col1+n_col2, 3),
                           dtype='uint8')
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
    rate = rospy.Rate(1)
    # get params
    rawfile = rospy.get_param('~rawfile', 'image.png')
    base, ext = os.path.splitext(rawfile)
    maskfile = rospy.get_param('~maskfile',
        '{base}_mask{ext}'.format(base=base, ext=ext))
    rospy.loginfo('rawfile: {raw}'.format(raw=rawfile))
    rospy.loginfo('maskfile: {mask}'.format(mask=maskfile))

    sm = SiftMatcherOneImg(rawfile, maskfile)
    while not rospy.is_shutdown():
        query_img, query_features = sm.query_img, sm.query_features
        train_img, train_features = sm.train_img, sm.train_features
        matches = SiftMatcherOneImg.find_match(query_features.descriptors,
                                               train_features.descriptors)
        rospy.loginfo('matches: {}'.format(len(matches)))
        # prepare output img
        matched_img = SiftMatcherOneImg.drawMatches(query_img,
            query_features.positions, train_img,
            train_features.positions, matches)
        cv2.putText(matched_img, 'matches: {}'.format(len(matches)),
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        # publish mathced_img
        bridge = cv_bridge.CvBridge()
        img_msg = bridge.cv2_to_imgmsg(matched_img, encoding='bgr8')
        img_msg.header.stamp = rospy.Time.now()
        sm.pub.publish(img_msg)

        rate.sleep()

if __name__ == '__main__':
    main()

