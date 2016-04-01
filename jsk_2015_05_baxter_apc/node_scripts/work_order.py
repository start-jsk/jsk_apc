#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json

import rospy
from jsk_2015_05_baxter_apc.msg import WorkOrder, WorkOrderArray

from bin_contents import get_bin_contents


def get_work_order(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)['work_order']
    for order in data:
        bin_ = order['bin'].split('_')[1].lower()  # bin_A -> a
        target_object = order['item']
        yield (bin_, target_object)


def get_sorted_work_order(json_file):
    """Sort work order to maximize the score"""
    bin_contents = get_bin_contents(json_file=json_file)
    bins, objects = zip(*bin_contents)
    bin_n_contents = dict(zip(bins, map(len, objects)))
    sorted_work_order = []
    work_order = dict(get_work_order(json_file=json_file))
    for bin_, _ in sorted(bin_n_contents.items(), key=lambda x:x[1]):
        sorted_work_order.append((bin_, work_order[bin_]))
    return sorted_work_order


def get_work_order_msg(json_file):
    work_order = get_sorted_work_order(json_file=json_file)
    msg = dict(left=WorkOrderArray(), right=WorkOrderArray())
    abandon_objects = [
        'genuine_joe_plastic_stir_sticks',
        'cheezit_big_original',
        'rolodex_jumbo_pencil_cup',
        'champion_copper_plus_spark_plug',
        'oreo_mega_stuf',
    ]
    for bin_, target_object in work_order:
        if target_object in abandon_objects:
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
    rate = rospy.Rate(rospy.get_param('rate', 1))
    while not rospy.is_shutdown():
        pub_left.publish(msg['left'])
        pub_right.publish(msg['right'])
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('work_order')
    main()
