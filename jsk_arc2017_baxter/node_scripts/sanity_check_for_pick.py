#!/usr/bin/env python

import os.path as osp
import sys

import rospy
import sensor_msgs.msg

from jsk_tools.sanity_lib import checkNodeState
from jsk_tools.sanity_lib import checkTopicIsPublished


def main():
    sys.stdout = sys.__stderr__

    # check node is exists
    nodes = [
    ]
    for node in nodes:
        checkNodeState(node, needed=True)

    # common for left/right hand camera
    topics = [
        '/json_saver/output/bin_contents',
        '/json_saver/output/json_dir',
    ]
    for topic in topics:
        if not checkTopicIsPublished(topic, timeout=5):
            return

    # for left/right hand camera
    topics = [
        # pick.launch
        'candidates_publisher/output/candidates',

        'weight_candidates_refiner/output/candidates/picked',
        'weight_candidates_refiner/output/candidates/placed',
    ]
    for side in ['left', 'right']:
        for topic in topics:
            topic = '/%s_hand_camera/%s' % (side, topic)
            if not checkTopicIsPublished(topic, timeout=30):
                return

    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    rospy.init_node('sanity_check_for_pick')
    main()
