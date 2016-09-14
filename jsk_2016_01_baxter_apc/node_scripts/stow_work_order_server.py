#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path as osp
import sys
import json
import rospy
import rospkg

from jsk_2015_05_baxter_apc.msg import WorkOrder, WorkOrderArray
import jsk_apc2016_common
from jsk_topic_tools.log_utils import jsk_logwarn


class StowWorkOrderServer():
    def __init__(self):
        rp = rospkg.RosPack()
        self.json_file = rospy.get_param('~json', None)
        self.create_output_json = rospy.get_param('~create_output_json', True)
        if self.create_output_json:
            self.output_json_file = osp.join(
                    rp.get_path('jsk_2016_01_baxter_apc'),
                    'output',
                    'output_' + osp.basename(self.json_file)
                    )
        else:
            self.output_json_file = self.json_file
        self.black_list = rospy.get_param('~black_list', [])
        self.volume_first = rospy.get_param('~volume_first', [])
        self.limit_volume = rospy.get_param('~limit_volume', 3000)
        if self.json_file is None:
            rospy.logerr('must set json file path to ~json')
            return
        if self.output_json_file is None:
            rospy.logerr('must set output json file path to ~output_json')
            return
        self.pub = {}
        self.pub['left'] = rospy.Publisher(
                '~left_hand',
                WorkOrderArray,
                queue_size=1
                )
        self.pub['right'] = rospy.Publisher(
                '~right_hand',
                WorkOrderArray,
                queue_size=1
                )
        self.object_data = jsk_apc2016_common.get_object_data()
        initial_bin_contents = jsk_apc2016_common.get_bin_contents(
                self.json_file
                )
        self.bin_point_dict = {
                k: self._get_point(v) for k, v in initial_bin_contents.items()
                }

    def main(self):
        duration = rospy.Duration(rospy.get_param('~duration', 0.1))
        rospy.Timer(duration, self.sort_work_order)
        rospy.spin()

    def _get_point(self, objects_list):
        if len(objects_list) > 4:
            return 20
        elif len(objects_list) > 2:
            return 15
        else:
            return 10

    def _get_volume(self, objects_list, target_object=''):
        total_volume = 0
        if target_object != '':
            objects_list.append(target_object)
        for object_name in objects_list:
            total_volume += [d['volume'] for d in self.object_data
                             if d['name'] == object_name][0]
        return total_volume

    def sort_by_volume(self, bin_list, bin_contents, target_object):
        bin_volume_list = [self._get_volume(bin_contents[bin_], target_object)
                           for bin_ in bin_list]
        bin_list = [bin_ for bin_, bin_volume in zip(bin_list, bin_volume_list)
                    if bin_volume < self.limit_volume]
        bin_list = sorted(
                bin_list,
                key=lambda bin_: self._get_volume(bin_contents[bin_])
                )
        return bin_list

    def sort_work_order(self, event):
        bin_contents = jsk_apc2016_common.get_bin_contents(self.output_json_file)
        msg = {}
        target_object = {}
        target_object['left'] = rospy.get_param('/left_hand/target_object')
        target_object['right'] = rospy.get_param('/right_hand/target_object')
        for arm in ['left', 'right']:
            if arm == 'left':
                sorted_bin_list = [x for x in 'abdegj']
            else:
                sorted_bin_list = [x for x in 'cfhikl']
            msg[arm] = WorkOrderArray()
            if target_object[arm] in self.black_list:
                order = WorkOrder(bin="tote", object=target_object[arm])
                msg[arm].array.append(order)
            else:
                if target_object[arm] in self.volume_first:
                    sorted_bin_list = sorted(
                            sorted_bin_list,
                            key=lambda bin_: self.bin_point_dict[bin_],
                            reverse=True
                            )
                    sorted_bin_list = self.sort_by_volume(
                            sorted_bin_list,
                            bin_contents,
                            target_object[arm]
                            )
                else:
                    sorted_bin_list = self.sort_by_volume(
                            sorted_bin_list,
                            bin_contents,
                            target_object[arm]
                            )
                    sorted_bin_list = sorted(
                            sorted_bin_list,
                            key=lambda bin_: self.bin_point_dict[bin_],
                            reverse=True
                            )
                for bin_ in sorted_bin_list:
                    order = WorkOrder(bin=bin_, object=target_object[arm])
                    msg[arm].array.append(order)
            self.pub[arm].publish(msg[arm])


if __name__ == '__main__':
    rospy.init_node('stow_work_order')
    stow_server = StowWorkOrderServer()
    stow_server.main()
