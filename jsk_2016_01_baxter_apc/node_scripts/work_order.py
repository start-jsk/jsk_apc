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
    """Sort work order to maximize the score.

    Args:
      - json_file (str): Json file path.
      - gripper (str): Gripper name. It must be in object_data.yaml.
      - object_data (list of dict): Object data that is defined
            in object_data.yaml.
    """
    bin_contents = jsk_apc2016_common.get_bin_contents(json_file=json_file)
    work_order = jsk_apc2016_common.get_work_order(json_file=json_file)

    # Only for APC2016, we defined graspabilities for each grippers
    if object_data is not None:
        sorted_bin_list = bin_contents.keys()

        def get_graspability(bin_):
            target_object = work_order[bin_]
            target_object_data = [data for data in object_data
                                  if data['name'] == target_object][0]
            graspability = target_object_data['graspability'][gripper]
            return graspability

        sorted_bin_list = sorted(sorted_bin_list, key=get_graspability)

    sorted_bin_list = sorted(sorted_bin_list,
                             key=lambda bin_: len(bin_contents[bin_]))
    sorted_work_order = [(bin_, work_order[bin_]) for bin_ in sorted_bin_list]
    return sorted_work_order


def get_work_order_msg(json_file, gripper, object_data=None):
    """Returns jsk_2015_05_baxter_apc/WorkOrderArray message.

    Args:
      - json_file (str): Json file path.
      - gripper (str): Gripper name. It must be in object_data.yaml.
      - object_data (list of dict): Object data that is defined
            in object_data.yaml.
    """
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
    # TODO(knorth55): Maybe will use max_weight to skip heavy objects
    max_weight = np.inf

    work_order = get_sorted_work_order(json_file, gripper, object_data)
    msg = dict(left=WorkOrderArray(), right=WorkOrderArray())
    bin_contents = jsk_apc2016_common.get_bin_contents(json_file=json_file)
    for bin_, target_object in work_order:
        if object_data is not None:
            # With object_data, use it for picking decision
            target_object_data = [data for data in object_data
                                  if data['name'] == target_object][0]
            if target_object_data['weight'] > max_weight:
                # abandon if the target object is too heavy
                jsk_logwarn(
                    'Skipping {obj} in {bin} because it is too heavy'
                    .format(obj=target_object_data['name'], bin=bin_))
                continue
        else:
            # Without object_data, use defined blacklist for picking decision
            if target_object in abandon_target_objects:
                # abandon if it is the target in the bin
                jsk_logwarn(
                    'Skipping {obj} in {bin} because it is listed to abandon'
                    .format(obj=target_object_data['name'], bin=bin_))
                continue
            if any(obj in bin_contents[bin_] for obj in abandon_bin_objects):
                # abandon if it is in the bin
                jsk_logwarn(
                    'Skipping {obj} in {bin} because there is objects'
                    'in {bin} listed to abandon'
                    .format(obj=target_object_data['name'], bin=bin_))
                continue
        if len(bin_contents[bin_]) > 5:  # Level3
            continue
        order = WorkOrder(bin=bin_, object=target_object)
        if bin_ in 'abdegj':
            msg['left'].array.append(order)
        elif bin_ in 'cfhikl':
            msg['right'].array.append(order)
        else:
            raise ValueError('Unsupported bin name: {0}'.format(bin_))
    return msg


def main():
    json_file = rospy.get_param('~json', None)
    gripper = rospy.get_param('~gripper', 'gripper2015')
    is_apc2016 = rospy.get_param('~is_apc2016', False)
    if json_file is None:
        rospy.logerr('must set json file path to ~json')
        return
    object_data = None
    if is_apc2016:
        object_data = jsk_apc2016_common.get_object_data()

    msg = get_work_order_msg(json_file, gripper, object_data)

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
