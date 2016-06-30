#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
import sys
import json
import rospy
from jsk_2015_05_baxter_apc.msg import WorkOrder, WorkOrderArray
import jsk_apc2016_common
from jsk_topic_tools.log_utils import jsk_logwarn

import numpy as np


def get_sorted_work_order(json_file, gripper, object_data):
    """Sort work order to maximize the score"""
    bin_contents = jsk_apc2016_common.get_bin_contents(json_file=json_file)
    work_order = jsk_apc2016_common.get_work_order(json_file=json_file)
    sorted_bin_list = bin_contents.keys()

    if object_data is not None:
        if all(gripper in x for x in [d['graspability'].keys() for d in object_data]):
            def get_graspability(bin_):
                target_object = work_order[bin_]
                target_object_data = [data for data in object_data
                                      if data['name'] == target_object][0]
                graspability = target_object_data['graspability'][gripper]
                return graspability
            sorted_bin_list = sorted(sorted_bin_list, key=get_graspability)
        else:
            jsk_logwarn('Not sorted by graspability')
            jsk_logwarn('Not all object_data have graspability key: {gripper}'
                        .format(gripper=gripper))
    sorted_bin_list = sorted(sorted_bin_list,
                             key=lambda bin_: len(bin_contents[bin_]))
    sorted_work_order = [(bin_, work_order[bin_]) for bin_ in sorted_bin_list]
    return sorted_work_order


def get_work_order_msg(json_file, gripper, max_weight, object_data=None):
    work_order = get_sorted_work_order(json_file, gripper, object_data)
    bin_contents = jsk_apc2016_common.get_bin_contents(json_file=json_file)
    if max_weight == -1:
        max_weight = np.inf
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
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # TODO: PLEASE FILL ABANDON BINS
    abandon_bins = ''
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    for bin_, target_object in work_order:
        if bin_ in abandon_bins:
            jsk_logwarn('Skipping bin {bin} because of user request'
                        .format(bin=bin_))
            continue
        if object_data is not None:
            target_object_data = [data for data in object_data
                                  if data['name'] == target_object][0]
            if target_object_data['weight'] > max_weight:
                jsk_logwarn('Skipping target {obj} in {bin_}: it exceeds max weight {weight} > {max_weight}'
                            .format(obj=target_object_data['name'], bin_=bin_,
                                    weight=target_object_data['weight'], max_weight=max_weight))
                continue
            if target_object_data['graspability'][gripper] > 3:
                jsk_logwarn('Skipping target {obj} in {bin_}: it exceeds graspability {grasp} > {max_grasp} with gripper {gripper}'
                            .format(obj=target_object_data['name'], bin_=bin_,
                                    grasp=target_object_data['graspability'][gripper],
                                    max_grasp=3,
                                    gripper=gripper))
                continue
        else:
            if target_object in abandon_target_objects:
                jsk_logwarn('Skipping target {obj} in {bin_}: it is listed as abandon target'
                            .format(obj=target_object, bin_=bin_))
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
    gripper = rospy.get_param('~gripper', 'gripper2016')
    max_weight = rospy.get_param('~max_weight', -1)
    if json_file is None:
        rospy.logerr('must set json file path to ~json')
        return
    object_data = None
    if is_apc2016:
        object_data = jsk_apc2016_common.get_object_data()

    msg = get_work_order_msg(json_file, gripper, max_weight, object_data)

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
