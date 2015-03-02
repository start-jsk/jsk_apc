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


def imgsift_client(img):
    img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
    img_msg.header.stamp = rospy.Time.now()
    client = rospy.ServiceProxy('/Feature0DDetect', Feature0DDetect)
    resp = client(img_msg)
    return resp.features

def find_match(query_img, query_features, train_img, train_features):
    query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
    query_pos = np.array(query_features.positions).reshape((-1, 2))
    query_des = np.array(query_features.descriptors).reshape((-1, 128))
    query_des = (query_des * 255).astype('uint8')
    train_img = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)
    train_pos = np.array(train_features.positions).reshape((-1, 2))
    train_des = np.array(train_features.descriptors).reshape((-1, 128))
    train_des = (train_des * 255).astype('uint8')

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(query_des, train_des, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    rospy.loginfo('number of good_matches: {}'.format(len(good_matches)))
    matched_img = drawMatches(query_img, query_pos, train_img, train_pos,
                              good_matches)
    cv2.putText(matched_img, 'good_matches: {}'.format(len(good_matches)),
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    return matched_img


def drawMatches(img1, pos1, img2, pos2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = pos1[img1_idx]
        (x2,y2) = pos2[img2_idx]
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)),
                 (255, 0, 0), 1)
    return out


def cb_imgfeature(msg):
    global query_features
    query_features = msg.features


def cb_img(msg):
    global query_img
    bridge = cv_bridge.CvBridge()
    query_img = bridge.imgmsg_to_cv2(msg)


if __name__ == '__main__':
    rospy.init_node('sift_matcher')
    query_features = query_img = None

    sub_imgfeature = rospy.Subscriber('/ImageFeature0D', ImageFeature0D,
                                      cb_imgfeature)
    sub_img = rospy.Subscriber('/image', Image, cb_img)
    pub = rospy.Publisher('~output', Image)

    filename = rospy.get_param('~filename', 'image.png')
    base, ext = os.path.splitext(filename)
    mask_file = rospy.get_param('~maskfile',
        '{base}_mask{ext}'.format(base=base, ext=ext))
    train_img = cv2.imread(filename)
    mask_img = cv2.imread(os.path.splitext(filename)[0]+'_mask.png')
    train_img = cv2.add(mask_img, train_img)
    bridge = cv_bridge.CvBridge()

    train_features = imgsift_client(train_img)

    while not rospy.is_shutdown():
        rospy.sleep(1.)

        if query_features is None or query_img is None:
            continue

        matched_img = find_match(query_img, query_features,
                                 train_img, train_features)
        img_msg = bridge.cv2_to_imgmsg(matched_img, encoding='bgr8')
        img_msg.header.stamp = rospy.Time.now()
        pub.publish(img_msg)

