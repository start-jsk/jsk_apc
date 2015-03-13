#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from collections import Counter

from sklearn.cluster import MiniBatchKMeans
from matplotlib.pyplot as plt

import rospy
from posedetection_msgs.msg import ImageFeature0D
from jsk_recognition_msgs.msg import ColorHistogram


def cb(msg):
    global feature_msg
    feature_msg = msg

rospy.init_node('bag_of_features')
sub = rospy.Subscriber('/ImageFeature0D', ImageFeature0D, cb)
pub = rospy.Publisher('/bag_of_features', ColorHistogram, queue_size=1)
rate = rospy.Rate(100)

while not rospy.is_shutdown():
    km = MiniBatchKMeans(n_clusters=1000)
    km.fit(descriptors)
    km.labels_

    msg = ColorHistogram()
    msg.header = feature_msg.header
    msg.histogram = histogram
    pub.publish(msg)

    rate.sleep()

