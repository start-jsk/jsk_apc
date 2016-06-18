#!/usr/bin/env python

import threading

import rospy
from jsk_apc2016_common.msg import BinInfo, BinInfoArray
from jsk_topic_tools import log_utils


class PublishTargetBinInfo(object):
    def __init__(self):
        self.lock = threading.Lock()

        self.bin_info_dict = {}
        self.bin_info_array_sub = rospy.Subscriber('~input/bin_info_array', BinInfoArray, self.callback)
        self.target_bin_info_pub = rospy.Publisher('~target_bin_info', BinInfo, queue_size=5)

        rate = rospy.get_param('~rate', 10)
        self.timer = rospy.Timer(rospy.Duration(1. / rate), self._publish)

    def _publish(self, event):
        if self.bin_info_dict == {}:
            return
        target_bin_name = rospy.get_param('~target_bin_name', '')
        if target_bin_name not in 'abcdefghijkl':
            rospy.logwarn('wrong target_bin_name')
            return
        if target_bin_name == '':
            log_utils.logwarn_throttle(10, 'target_bin_name is empty string. This shows up every 10 seconds.')
            return

        with self.lock:
            self.bin_info_dict[target_bin_name].header.seq = (
                self.bin_info_dict[target_bin_name].header.seq + 1)
        target_bin_info = self.bin_info_dict[target_bin_name]
        target_bin_info.header.stamp = rospy.Time.now()
        self.target_bin_info_pub.publish(target_bin_info)

    def bin_info_array_to_dict(self, bin_info_array):
        bin_info_dict = {}
        for bin_ in bin_info_array.array:
            bin_info_dict[bin_.name] = bin_
        return bin_info_dict

    def callback(self, bin_info_array):
        with self.lock:
            self.bin_info_dict = self.bin_info_array_to_dict(bin_info_array)


if __name__ == '__main__':
    rospy.init_node('publish_target_bin_info')
    publish_target_bin_info = PublishTargetBinInfo()
    rospy.spin()
