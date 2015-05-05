#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import rospy

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
        self.sub = rospy.Subscriber('/bof_object_matcher/output',
                                    ObjectRecognition, self._cb_bof)
        self.pub = rospy.Publisher('~output', ObjectRecognition, queue_size=1)
        self.bof_data = None

    def _init_bin_contents(self, json_file):
        bin_contents = get_bin_contents(json_file)
        self.bin_contents = dict(bin_contents)

    def _init_work_order(self, json_file):
        work_order = get_work_order(json_file)
        self.work_order = dict(work_order)

    def _cb_bof(self, msg):
        objects_proba = dict(zip(msg.candidates, msg.probabilities))
        self.bof_data = (msg.header.stamp, objects_proba)

    def spin_once(self):
        if self.bof_data is None:
            return
        stamp, objects_proba = self.bof_data
        target_bin = rospy.get_param('/target', None)
        if target_bin is None:
            return
        target_object = self.work_order[target_bin]
        candidates = self.bin_contents[target_bin]
        proba = [(c, objects_proba[c]) for c in candidates]
        matched = sorted(proba, key=lambda x: x[1])[-1][0]
        # compose msg
        msg = ObjectRecognition()
        msg.header.stamp = stamp
        msg.matched = matched
        msg.probability = objects_proba[matched]
        msg.candidates = candidates
        msg.probabilities = [objects_proba[c] for c in candidates]
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