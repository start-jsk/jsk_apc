#!/usr/bin/env python

import numpy as np

from jsk_2015_05_baxter_apc.msg import ObjectRecognition
import jsk_apc2016_common
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_topic_tools import ConnectionBasedTransport
from jsk_topic_tools.log_utils import jsk_logwarn
import rospy


class ApplyBinContentsHint(ConnectionBasedTransport):

    """Use bin contents info of target bin to improve object recognition"""

    def __init__(self):
        super(self.__class__, self).__init__()
        json_file = rospy.get_param('~json')
        self.bin_contents = jsk_apc2016_common.get_bin_contents(json_file)
        self.pub = self.advertise('~output', ObjectRecognition, queue_size=1)

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', ClassificationResult,
                                    self._apply, queue_size=1)

    def unsubscribe(self):
        self.sub.unregister()

    def _apply(self, msg):
        target_bin = rospy.get_param('target_bin', '')
        if (not target_bin) or (target_bin not in 'abcdefghijkl'):
            return

        # get candidates probabilities
        if target_bin not in self.bin_contents:
            rospy.logerr("target_bin '{0}' is not found in json: {1}"
                         .format(target_bin, self.bin_contents))
            return

        candidates = self.bin_contents[target_bin] + ['no_object']
        label_to_proba = dict(zip(msg.target_names, msg.probabilities))
        candidates_proba = [label_to_proba[label] for label in candidates]
        candidates_proba = np.array(candidates_proba)
        candidates_proba = candidates_proba / candidates_proba.sum()

        # compose output message
        top_index = np.argmax(candidates_proba)
        out_msg = ObjectRecognition(
            header=msg.header,
            matched=candidates[top_index],
            probability=candidates_proba[top_index],
            candidates=candidates,
            probabilities=candidates_proba,
        )
        self.pub.publish(out_msg)


if __name__ == '__main__':
    rospy.init_node('apply_bin_contents_hint')
    app = ApplyBinContentsHint()
    rospy.spin()
