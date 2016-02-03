#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy as np
import yaml

import message_filters
import rospy

from jsk_recognition_msgs.msg import ClassificationResult
from jsk_topic_tools import ConnectionBasedTransport


class BoostObjectRecognition(ConnectionBasedTransport):

    def __init__(self):
        super(BoostObjectRecognition, self).__init__()
        weight_yaml = rospy.get_param('~weight', None)
        if weight_yaml is None:
            rospy.logerr('must set weight yaml file path to ~weight')
            return
        with open(weight_yaml) as f:
            self.weight = yaml.load(f)
        self.pub = self.advertise(
            '~output', ClassificationResult, queue_size=1)

    def subscribe(self):
        self.sub_bof = message_filters.Subscriber(
            '~input/bof', ClassificationResult)
        self.sub_ch = message_filters.Subscriber(
            '~input/ch', ClassificationResult)
        queue_size = rospy.get_param('~queue_size', 100)
        if rospy.get_param('~approximate_sync', False):
            sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_bof, self.sub_ch], queue_size=queue_size,
                slop=1)
        else:
            sync = message_filters.TimeSynchronizer(
                [self.sub_bof, self.sub_ch], queue_size=queue_size)
        sync.registerCallback(self._apply)

    def unsubscribe(self):
        self.sub_bof.unregister()
        self.sub_ch.unregister()

    def _apply(self, bof_msg, ch_msg):
        target_names = bof_msg.target_names
        assert target_names == ch_msg.target_names

        N_label = len(target_names)
        bof_proba = np.array(bof_msg.probabilities).reshape((-1, N_label))
        ch_proba = np.array(ch_msg.probabilities).reshape((-1, N_label))
        bof_weight = np.array([self.weight[n]['bof'] for n in target_names])
        ch_weight = np.array([self.weight[n]['color'] for n in target_names])
        y_proba = (bof_weight * bof_proba) + (ch_weight * ch_proba)

        # verification result for debug
        y_pred = np.argmax(y_proba, axis=-1)
        target_names = np.array(target_names)
        label_proba = [p[i] for p, i in zip(y_proba, y_pred)]
        # compose msg
        msg = ClassificationResult()
        msg.header = bof_msg.header
        msg.labels = y_pred
        msg.label_names = target_names[y_pred]
        msg.label_proba = label_proba
        msg.probabilities = y_proba.reshape(-1)
        msg.classifier = '<jsk_2015_05_baxter_apc.BoostObjectRecognition>'
        msg.target_names = target_names
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('boost_object_recognition')
    boost_or = BoostObjectRecognition()
    rospy.spin()
