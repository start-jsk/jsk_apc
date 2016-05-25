#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
import sys
import json
import rospy
from jsk_2015_05_baxter_apc.msg import WorkOrder, WorkOrderArray
import jsk_apc2016_common
from jsk_topic_tools.log_utils import jsk_logwarn


def get_sorted_work_order(json_file):
    """Sort work order to maximize the score"""
    bin_contents = jsk_apc2016_common.get_bin_contents(json_file=json_file)
    work_order = jsk_apc2016_common.get_work_order(json_file=json_file)
    bin_n_contents = dict(map(lambda (bin_, objects): (bin_, len(objects)), bin_contents.iteritems()))
    sorted_work_order = []
    for bin_, n_contents in sorted(bin_n_contents.items(), key=lambda x: x[1]):
        sorted_work_order.append((bin_, work_order[bin_]))
    return sorted_work_order


def get_work_order_msg(json_file, object_data=None):
    work_order = get_sorted_work_order(json_file=json_file)
    bin_contents = jsk_apc2016_common.get_bin_contents(json_file=json_file)
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
    for bin_, target_object in work_order:
        if object_data is not None:
            target_object_data = [data for data in object_data
                                  if data['name'] == target_object][0]
        else:
            if target_object in abandon_target_objects:
                jsk_logwarn('Skipping target {obj} in {bin_}: it is listed as abandon target'
                            .format(obj=target_object['name'], bin_=bin_))
                continue
            if any(bin_object in abandon_bin_objects for bin_object in bin_contents[bin_]):
                jsk_logwarn('Skipping {bin_}: this bin contains abandon objects'.format(bin_=bin_))
                continue
        if len(bin_contents[bin_]) > 5:  # Level3
            jsk_logwarn('Skipping {bin_}: this bin contains more than 5 objects'.format(bin_=bin_))
            continue
        order = WorkOrder(bin=bin_, object=target_object)
        if bin_ in 'abdegj':
            msg['left'].array.append(order)
        elif bin_ in 'cfhikl':
            msg['right'].array.append(order)
    return msg


def main():
    json_file = rospy.get_param('~json', None)
    is_apc2016 = rospy.get_param('~is_apc2016', True)
    if json_file is None:
        rospy.logerr('must set json file path to ~json')
        return
    object_data = None
    if is_apc2016:
        object_data = jsk_apc2016_common.get_object_data()

    msg = get_work_order_msg(json_file, object_data)

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
