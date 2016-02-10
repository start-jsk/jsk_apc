#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division
import yaml

import rospy

import numpy as np

from bin_contents import get_bin_contents
from work_order import get_work_order

import jsk_apc2015_common
from jsk_2015_05_baxter_apc.msg import ObjectRecognition
from jsk_recognition_msgs.msg import ClassificationResult


class ObjectVerification(object):
    def __init__(self):
        json_file = rospy.get_param('~json', None)
        weight_yaml = rospy.get_param('~weight', None)
        if json_file is None:
            rospy.logerr('must set json file path to ~json')
            return
        if weight_yaml is None:
            rospy.logerr('must set weight yaml file path to ~weight')
            return
        self.bof_data = None
        self.cfeature = None
        self._init_weight(weight_yaml)
        self._init_bin_contents(json_file)
        self._init_work_order(json_file)
        self.bof_sub = rospy.Subscriber('~input/bof',
                                        ClassificationResult,
                                        self._cb_bof)
        self.cfeature_sub = rospy.Subscriber('~input/color_hist',
                                             ClassificationResult,
                                             self._cb_cfeature)
        self.pub = rospy.Publisher('~output', ObjectRecognition, queue_size=1)
        self.pub_debug = rospy.Publisher('~debug', ObjectRecognition,
                                         queue_size=1)

    def _init_weight(self, yaml_file):
        with open(yaml_file) as f:
            weight = yaml.load(f)
        self.weight = weight

    def _init_bin_contents(self, json_file):
        bin_contents = get_bin_contents(json_file)
        self.bin_contents = dict(bin_contents)

    def _init_work_order(self, json_file):
        work_order = get_work_order(json_file)
        self.work_order = dict(work_order)

    def _cb_bof(self, msg):
        n_targets = len(msg.target_names)
        objects_proba = dict(zip(msg.target_names,
                                 msg.probabilities[:n_targets]))
        self.bof_data = (msg.header.stamp, objects_proba)

    def _cb_cfeature(self, msg):
        n_targets = len(msg.target_names)
        objects_proba = dict(zip(msg.target_names,
                                 msg.probabilities[:n_targets]))
        self.cfeature = (msg.header.stamp, objects_proba)

    def spin_once(self):
        if self.bof_data is None or self.cfeature is None:
            return
        stamp, bof_objects_proba = self.bof_data
        stamp, cfeature_objects_proba = self.cfeature
        weight = self.weight

        target_bin = rospy.get_param('target_bin', None)

        object_list = jsk_apc2015_common.get_object_list()
        all_proba = [
            (o,
             (weight[o]['bof'] * bof_objects_proba[o]) +
             (weight[o]['color'] * cfeature_objects_proba[o])
             ) for o in object_list
            ]

        # verification result for debug
        candidates = self.bin_contents.get(target_bin, None)
        if candidates is None:
            candidates = object_list
        matched = sorted(all_proba, key=lambda x: x[1])[-1][0]
        # compose msg
        msg = ObjectRecognition()
        msg.header.stamp = stamp
        msg.matched = matched
        msg.probability = dict(all_proba)[matched] / sum(dict(all_proba).values())
        msg.candidates = candidates
        msg.probabilities = np.array([dict(all_proba)[c] for c in candidates])
        msg.probabilities /= msg.probabilities.sum()
        self.pub_debug.publish(msg)

        # verification result with json target
        if target_bin is None or target_bin == '':
            return
        proba = [
            (c,
             (weight[c]['bof'] * bof_objects_proba[c]) +
             (weight[c]['color'] * cfeature_objects_proba[c])
             ) for c in candidates
            ]
        matched = sorted(proba, key=lambda x: x[1])[-1][0]
        # compose msg
        msg = ObjectRecognition()
        msg.header.stamp = stamp
        msg.matched = matched
        msg.probability = dict(proba)[matched] / sum(dict(proba).values())
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
