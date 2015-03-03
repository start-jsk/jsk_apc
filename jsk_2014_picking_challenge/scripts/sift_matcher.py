#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import gzip
import cPickle as pickle

import numpy as np
import yaml

import rospy
from posedetection_msgs.msg import ImageFeature0D
from jsk_2014_picking_challenge.srv import ObjectMatch, ObjectMatchResponse

from sift_matcher_oneimg import SiftMatcherOneImg


class SiftMatcher(object):
    def __init__(self):
        # load object list
        dirname = os.path.dirname(os.path.abspath(__file__))
        ymlfile = os.path.join(dirname, '../data/object_list.yml')
        self.object_list = yaml.load(open(ymlfile))

        rospy.loginfo('Loading sift data')
        self.all_siftdata = self._load_all_siftdata()

        rospy.Service('/semi/sift_matcher', ObjectMatch, self._cb_matcher)
        sub_imgfeature = rospy.Subscriber('/ImageFeature0D', ImageFeature0D,
                                          self._cb_imgfeature)
        rospy.loginfo('Ready to reseive match request')

    def _cb_matcher(self, req):
        probs = self._get_object_probability(req.objects)
        return ObjectMatchResponse(probabilities=probs)

    def _cb_imgfeature(self, msg):
        self.query_features = msg.features

    def _get_object_probability(self, obj_names):
        """Get object match probabilities"""
        query_features = self.query_features
        n_matches = []
        for obj_name in obj_names:
            if obj_name not in self.object_list:
                n_matches.append(0)
                continue
            # find best match in train features
            siftdata = self.all_siftdata.get(obj_name, None)
            if siftdata is None:  # does not exists data file
                n_matches.append(0)
                continue
            train_matches = []
            for train_des in siftdata['descriptors']:
                matches = SiftMatcherOneImg.find_match(
                    query_features.descriptors, train_des)
                train_matches.append(len(matches))
            n_matches.append(max(train_matches))  # best match
        n_matches = np.array(n_matches)
        return n_matches / n_matches.max()

    def _load_all_siftdata(self):
        """Load sift data of all objects"""
        object_list = self.object_list
        all_siftdata = {obj_name: self.load_siftdata(obj_name)
                        for obj_name in object_list}
        return all_siftdata

    @staticmethod
    def load_siftdata(obj_name):
        """Load sift data from pkl file"""
        dirname = os.path.dirname(os.path.abspath('__file__'))
        datafile = os.path.join(dirname, '../data/siftdata',
            obj_name+'.pkl.gz')
        if not os.path.exists(datafile):
            return  # does not exists
        with gzip.open(datafile, 'rb') as f:
            return pickle.load(f)


def main():
    rospy.init_node('sift_matcher')
    sm = SiftMatcher()
    rospy.spin()


if __name__ == '__main__':
    main()

