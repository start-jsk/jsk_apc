#!/usr/bin/env python

import dynamic_reconfigure.server
from jsk_topic_tools import ConnectionBasedTransport
import json
import os.path as osp
import rospy
from std_msgs.msg import String

from jsk_arc2017_common.cfg import CandidatesPublisherConfig
from jsk_recognition_msgs.msg import Label
from jsk_recognition_msgs.msg import LabelArray


class CandidatesPublisher(ConnectionBasedTransport):

    def __init__(self):
        super(CandidatesPublisher, self).__init__()
        self.pub = self.advertise(
            '~output/candidates', LabelArray, queue_size=1)
        self.srv = dynamic_reconfigure.server.Server(
            CandidatesPublisherConfig, self._config_cb)
        self.label_names = rospy.get_param('~label_names')

    def subscribe(self):
        self.sub = rospy.Subscriber('~input/json_dir', String, self._cb)

    def unsubscribe(self):
        self.sub.unregister()

    def _config_cb(self, config, level):
        self.target_location = config.target_location
        return config

    def _cb(self, msg):
        json_dir = msg.data
        if not osp.isdir(json_dir):
            rospy.logfatal_throttle(
                10, 'Input json_dir is not directory: %s' % json_dir)
            return
        filename = osp.join(json_dir, 'item_location_file.json')
        if osp.exists(filename):
            with open(filename) as location_f:
                data = json.load(location_f)

            bin_contents = {}
            for bin_ in data['bins']:
                bin_contents[bin_['bin_id']] = bin_['contents']
            tote_contents = data['tote']['contents']

            if self.target_location[:3] == 'bin':
                contents = bin_contents[self.target_location[4]]
            elif self.target_location == 'tote':
                contents = tote_contents
            else:
                return
            candidates_fixed = [l for l in self.label_names
                                if l.startswith('__')]
            candidates = candidates_fixed + contents
            label_list = [self.label_names.index(x) for x in candidates]
            label_list = sorted(label_list)
            labels = []
            for label in label_list:
                label_msg = Label()
                label_msg.id = label
                label_msg.name = self.label_names[label]
                labels.append(label_msg)
            msg = LabelArray()
            msg.labels = labels
            msg.header.stamp = rospy.Time.now()
            self.pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('candidates_publisher')
    candidates_publisher = CandidatesPublisher()
    rospy.spin()
