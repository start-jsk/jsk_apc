#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import rospy

import numpy as np

from bin_contents import get_bin_contents
from work_order import get_work_order
from jsk_2014_picking_challenge.msg import ObjectRecognition


class ObjectVerification(object):
    def __init__(self):
        json_file = rospy.get_param('~json', None)
        if json_file is None:
            rospy.logerr('must set json file path to ~json')
            return
        self._init_bin_contents(json_file)
        self._init_work_order(json_file)
        self.bof_sub = rospy.Subscriber('/bof_object_matcher/output',
                                    ObjectRecognition, self._cb_bof)
        self.cfeature_sub = rospy.Subscriber('/color_object_matcher/output',
                                             ObjectRecognition, self._cb_cfeature)
        self.pub = rospy.Publisher('~output', ObjectRecognition, queue_size=1)
        self.bof_data = None
        self.cfeature = None

    def _init_bin_contents(self, json_file):
        bin_contents = get_bin_contents(json_file)
        self.bin_contents = dict(bin_contents)

    def _init_work_order(self, json_file):
        work_order = get_work_order(json_file)
        self.work_order = dict(work_order)

    def _cb_bof(self, msg):
        objects_proba = dict(zip(msg.candidates, msg.probabilities))
        self.bof_data = (msg.header.stamp, objects_proba)

    def _cb_cfeature(self, msg):
        objects_proba = dict(zip(msg.candidates, msg.probabilities))
        self.cfeature = (msg.header.stamp, objects_proba)

    def spin_once(self):
        if self.bof_data is None or self.cfeature is None:
            return
        stamp, bof_objects_proba = self.bof_data
        stamp, cfeature_objects_proba = self.cfeature

        target_bin = rospy.get_param('/target', None)
        if target_bin is None:
            return
        candidates = self.bin_contents[target_bin]
        proba = [(c, bof_objects_proba[c] + cfeature_objects_proba[c]) for c in candidates]

        matched = sorted(proba, key=lambda x: x[1])[-1][0]
        # compose msg
        msg = ObjectRecognition()
        msg.header.stamp = stamp
        msg.matched = matched
        msg.probability = dict(proba)[matched]
        msg.candidates = candidates
        msg.probabilities = np.array([dict(proba)[c] for c in candidates])
        msg.probabilities /= msg.probabilities.sum()
        self.pub.publish(msg)

    def spin(self):
        rate = rospy.Rate(rospy.get_param('rate', 10))
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('object_verification')
    verification = ObjectVerification()
    verification.spin()
