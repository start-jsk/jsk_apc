#!/usr/bin/env python
#
import os
import itertools

import cv2
import numpy as np

import rospy
import cv_bridge
from sensor_msgs.msg import Image
from posedetection_msgs.msg import ImageFeature0D
from posedetection_msgs.srv import Feature0DDetect


class SiftMatcherOneImage(object):
    def __init__(self, rawfile, maskfile):
        # Subscribers
        sub_imgfeature = rospy.Subscriber('/ImageFeature0D', ImageFeature0D,
                                          self.cb_imgfeature)
        sub_img = rospy.Subscriber('/image', Image, self.cb_img)
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

    def cb_imgfeature(self, msg):
        """Callback function of Subscribers to listen ImageFeature0D"""
        self.query_features = msg.features

    def cb_img(self, msg):
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
    def find_match(query_img, query_features, train_img, train_features):
        """Find match points of query and train images"""
        # parepare to match keypoints
        query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
        query_pos = np.array(query_features.positions).reshape((-1, 2))
        query_des = np.array(query_features.descriptors).reshape((-1, 128))
        query_des = (query_des * 255).astype('uint8')
        train_img = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)
        train_pos = np.array(train_features.positions).reshape((-1, 2))
        train_des = np.array(train_features.descriptors).reshape((-1, 128))
        train_des = (train_des * 255).astype('uint8')
        # find good match points
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(query_des, train_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75*n.distance]
        rospy.loginfo('good_matches: {}'.format(len(good_matches)))
        # prepare output img
        matched_img = SiftMatcherOneImage.drawMatches(query_img, query_pos,
            train_img, train_pos, good_matches)
        cv2.putText(matched_img, 'good_matches: {}'.format(len(good_matches)),
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        return matched_img

    @staticmethod
    def drawMatches(img1, pos1, img2, pos2, matches):
        """Draw match points for two images"""
        n_row1, n_col1 = img1.shape[:2]
        n_row2, n_col2 = img2.shape[:2]
        # parepare output img
        img_out = np.zeros((max([n_row1,n_row2]), n_col1+n_col2, 3),
                           dtype='uint8')
        img_out[:n_row1, :n_col1, :] = np.dstack(3*[img1])
        img_out[:n_row2, n_col1:n_col1+n_col2, :] = np.dstack(3*[img2])
        for mat in matches:
            # draw and connect match points
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
            x1, y1 = pos1[img1_idx]
            x2, y2 = pos2[img2_idx]
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

    sm = SiftMatcherOneImage(rawfile, maskfile)
    while not rospy.is_shutdown():
        matched_img = sm.find_match(sm.query_img, sm.query_features,
                                    sm.train_img, sm.train_features)
        # publish mathced_img
        bridge = cv_bridge.CvBridge()
        img_msg = bridge.cv2_to_imgmsg(matched_img, encoding='bgr8')
        img_msg.header.stamp = rospy.Time.now()
        sm.pub.publish(img_msg)

        rate.sleep()

if __name__ == '__main__':
    main()

