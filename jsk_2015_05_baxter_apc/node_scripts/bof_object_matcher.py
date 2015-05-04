#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division
import os
import gzip
import cPickle as pickle

import numpy as np
from sklearn.preprocessing import normalize

import rospy
from posedetection_msgs.msg import ImageFeature0D
from jsk_2014_picking_challenge.msg import ObjectRecognition

from bag_of_features import BagOfFeatures
from common import ObjectMatcher, get_data_dir, get_object_list


class BofObjectMatcher(ObjectMatcher):
    def __init__(self):
        super(BofObjectMatcher, self).__init__('/bof_object_matcher')
        rospy.Subscriber('~input', ImageFeature0D, self._cb_imgfeature)
        self._pub_recog = rospy.Publisher('~output', ObjectRecognition,
                                          queue_size=1)
        self._init_bof()
        self._init_clf()
        self.query_features = ImageFeature0D().features

    def _init_bof(self):
        data_dir = get_data_dir()
        bof_path = os.path.join(data_dir, 'bof_data/bof.pkl.gz')
        with gzip.open(bof_path) as f:
            bof = pickle.load(f)
        self.bof = bof

    def _init_clf(self):
        data_dir = get_data_dir()
        clf_path = os.path.join(data_dir, 'bof_data/lgr.pkl.gz')
        with gzip.open(clf_path) as f:
            lgr = pickle.load(f)
        self.clf = lgr

    def _cb_imgfeature(self, msg):
        """Callback function of Subscribers to listen ImageFeature0D"""
        self.query_features = msg.features

    def match(self, obj_names):
        stamp = rospy.Time.now()
        while self.query_features.header.stamp < stamp:
            rospy.sleep(0.3)
        descs = np.array(self.query_features.descriptors)
        X = self.bof.transform([descs])
        normalize(X, copy=False)
        object_list = get_object_list()
        obj_indices = [object_list.index(o) for o in obj_names]
        obj_probs = self.clf.predict_proba(X)[0][obj_indices]
        return obj_probs / obj_probs.sum()

    def predict_now(self):
        query_features = self.query_features
        if not len(query_features.descriptors) > 0:
            return
        descs = np.array(query_features.descriptors)
        X = self.bof.transform([descs])
        normalize(X, copy=False)
        object_list = get_object_list()
        proba = self.clf.predict_proba(X)[0]
        matched_idx = np.argmax(proba)
        # prepare message
        res = ObjectRecognition()
        res.header.stamp = query_features.header.stamp
        res.matched = object_list[matched_idx]
        res.probability = proba[matched_idx]
        res.candidates = object_list
        res.probabilities = proba
        return res

    def spin_once(self):
        res = self.predict_now()
        if res is None:
            return
        self._pub_recog.publish(res)

    def spin(self):
        rate = rospy.Rate(rospy.get_param('rate', 1))
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()


def main():
    rospy.init_node('bof_object_matcher')
    matcher = BofObjectMatcher()
    matcher.spin()


if __name__ == '__main__':
    main()

