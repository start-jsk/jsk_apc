#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import jsk_apc2015_common
from jsk_recognition_msgs.msg import ClassificationResult
import rospy
import json

class ConsiderJson():

    def __init__(self):

        json_data = rospy.get_param('~json_data', default='../json/apc_pick.json')
        bin_n = rospy.get_param('~bin', default='bin_E')
        if json_data is None:
            quit()
        with open(json_data, 'rb') as f:
            bins = json.load(f)['bin_contents']
        self.bin_contents = bins[bin_n]
        ####
        self.bin_contents = ['champion_copper_plus_spark_plug',
                             'cheezit_big_original',
                             'rolodex_jumbo_pencil_cup',
                             'paper_mate_12_count_mirado_black_warrior',
                             'mead_index_cards',
                             'stanley_66_052']

        ####
        self.target_names = jsk_apc2015_common.get_object_list()
        self.pub = rospy.Publisher('~output', ClassificationResult,queue_size=10)
        self.sub = rospy.Subscriber('~input', ClassificationResult,self._predict)


    def _predict(self, msg):

        self.target_names = msg.target_names
        n_target = len(self.target_names)
        msg_proba = msg.probabilities
        msg_proba = np.asarray(msg_proba)
        msg_proba = msg_proba.reshape((len(msg_proba)/n_target,n_target))
        msg_labels = msg.labels
        msg_labels = np.asarray(msg_labels)
        msg_label_names = msg.label_names

        for i in range(len(msg_label_names)):
            label_name = msg_label_names[i]
            label = msg_labels[i]
            rospy.logerr(label_name)
            while not label_name in self.bin_contents:
                msg_proba[i][label] = 0.0
                label = np.argmax(msg_proba[i])
                label_name = self.target_names[label]
            rospy.logerr(label_name)
            msg_label_names[i] = label_name
            msg_labels[i] = label
        label_proba = [p[i] for p, i in zip(msg_proba, msg_labels)]

        # prepare message
        res=ClassificationResult()
        res.header = msg.header
        res.labels = msg_labels
        res.label_names = msg_label_names
        res.label_proba = label_proba
        res.probabilities = msg_proba.reshape(-1)
        res.classifier = '<Json Knows Everything>'
        res.target_names = self.target_names
        self.pub.publish(res)


if __name__ == "__main__":
    rospy.init_node('consider_json')
    ConsiderJson()
    rospy.spin()
