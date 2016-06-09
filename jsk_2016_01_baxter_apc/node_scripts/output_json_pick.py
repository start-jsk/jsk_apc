#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path as osp
import json
import rospy
import rospkg
import jsk_apc2016_common
from jsk_topic_tools.log_utils import jsk_logwarn, jsk_loginfo


class OutputJsonPick():
    def __init__(self):
        rp = rospkg.RosPack()
        input_path = rospy.get_param('~json', None)
        if input_path is None:
            jsk_logwarn('must set json file path to param ~json')
            return
        with open(input_path, 'r') as f:
            self.output_data = json.load(f)
        self.tote_contents = self.output_data['tote_contents']
        self.bin_contents = self.output_data['bin_contents']
        self.work_order = jsk_apc2016_common.get_work_order(input_path)
        self.output_path = osp.join(
                rp.get_path('jsk_2016_01_baxter_apc'),
                'output',
                'output_' + osp.basename(input_path))
        self.arm_state = {}
        self.arm_target_bin = {}
        self.finished_bin_list = []
        with open(self.output_path, 'w') as f:
            json.dump(self.output_data, f, sort_keys=True, indent=4)

    def main(self):
        duration = rospy.Duration(rospy.get_param('~duration', 1))
        timer = rospy.Timer(duration, self.update_json)
        rospy.spin()

    def update_json(self, event):
        self.update_param()
        is_updated = self.update_data()
        if is_updated:
            self.output_data['tote_contents'] = self.tote_contents
            self.output_data['bin_contents'] = self.bin_contents
            with open(self.output_path, 'w') as f:
                json.dump(self.output_data, f, sort_keys=True, indent=4)

    def update_param(self):
        self.arm_state['right'] = rospy.get_param('/right_hand/state')
        self.arm_target_bin['right'] = \
            rospy.get_param('/right_hand/target_bin')
        self.arm_state['left'] = rospy.get_param('/left_hand/state')
        self.arm_target_bin['left'] = \
            rospy.get_param('/left_hand/target_bin')

    def update_data(self):
        is_updated = False
        for arm in ['right', 'left']:
            target_bin = self.arm_target_bin[arm]
            if target_bin not in self.finished_bin_list \
                    and target_bin in 'abcdefghijkl' \
                    and self.arm_state[arm] == "place_object":
                target_object = self.work_order[target_bin]
                self.tote_contents.append(target_object)
                target_bin_ = 'bin_' + target_bin.upper()
                self.bin_contents[target_bin_].remove(target_object)
                self.finished_bin_list.append(target_bin)
                jsk_loginfo('bin {}: object {} has moved to tote'
                            .format(target_bin, target_object))
                is_updated = True
        return is_updated

if __name__ == '__main__':
    rospy.init_node('output_json_pick')
    ojp = OutputJsonPick()
    ojp.main()
