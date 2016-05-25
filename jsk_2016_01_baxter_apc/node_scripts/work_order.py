#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
import sys
import json
import rospy
from jsk_2015_05_baxter_apc.msg import WorkOrder, WorkOrderArray
from jsk_apc2016_common import get_bin_contents, get_work_order, get_object_data



def get_sorted_work_order(json_file):
    """Sort work order to maximize the score"""
    bin_contents = get_bin_contents(json_file=json_file)
    bin_n_contents = dict(map(lambda (bin_, objects): (bin_, len(objects)), bin_contents.iteritems()))
    sorted_work_order = []
    work_order = get_work_order(json_file=json_file)
    for bin_, n_contents in sorted(bin_n_contents.items(), key=lambda x: x[1]):
        sorted_work_order.append((bin_, work_order[bin_]))
    return sorted_work_order


def get_work_order_msg(json_file):
    max_weight = rospy.get_param('~max_weight', -1)
    apc2016 = rospy.get_param('~apc2016', False)

    work_order = get_sorted_work_order(json_file=json_file)
    msg = dict(left=WorkOrderArray(), right=WorkOrderArray())
    abandon_target_objects = [
        'genuine_joe_plastic_stir_sticks',
        'cheezit_big_original',
        'rolodex_jumbo_pencil_cup',
        'champion_copper_plus_spark_plug',
        'oreo_mega_stuf',
    ]
    abandon_bin_objects = [
        'rolodex_jumbo_pencil_cup',
        'oreo_mega_stuf'
    ]
    bin_contents = get_bin_contents(json_file=json_file)
    object_data = get_object_data()
    for bin_, target_object in work_order:
        if apc2016:
            target_object_data = [data for data in object_data if data['name'] == target_object][0]
            if target_object_data:
                if target_object_data['weight'] > max_weight and max_weight != -1:
                    continue
            else:
                continue
        else:
            if target_object in abandon_target_objects:
                continue
            if [bin_object for bin_object in bin_contents[bin_] if bin_object in abandon_bin_objects]:
                continue
        if len(bin_contents[bin_]) > 5:  # Level3
            continue
        if bin_ in 'abdegj':
            msg['left'].array.append(WorkOrder(bin=bin_, object=target_object))
        elif bin_ in 'cfhikl':
            msg['right'].array.append(WorkOrder(bin=bin_, object=target_object))
    return msg


def main():
    json_file = rospy.get_param('~json', None)
    if json_file is None:
        rospy.logerr('must set json file path to ~json')
        return

    msg = get_work_order_msg(json_file)

    pub_left = rospy.Publisher('~left_hand',
                               WorkOrderArray,
                               queue_size=1)
    pub_right = rospy.Publisher('~right_hand',
                                WorkOrderArray,
                                queue_size=1)
    rate = rospy.Rate(rospy.get_param('~rate', 1))
    while not rospy.is_shutdown():
        pub_left.publish(msg['left'])
        pub_right.publish(msg['right'])
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('work_order')
    main()
