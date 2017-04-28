#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
import sys
import json
import rospy
from jsk_2015_05_baxter_apc.msg import WorkOrder, WorkOrderArray
import jsk_apc2016_common
from jsk_topic_tools.log_utils import jsk_logwarn
from jsk_apc2016_common.srv import UpdateTarget, UpdateTargetResponse

import numpy as np


class WorkOrderServer(object):

    def __init__(self):
        self.json_file = rospy.get_param('~json', None)
        self.is_apc2016 = rospy.get_param('~is_apc2016', True)
        self.gripper = rospy.get_param('~gripper', 'gripper2016')
        self.max_weight = rospy.get_param('~max_weight', -1)
        if self.json_file is None:
            rospy.logerr('must set json file path to ~json')
            return
        data = json.load(open(self.json_file))
        self.work_order = {}
        work_order_list = []
        for order in data['work_order']:
            bin_ = order['bin'].split('_')[1].lower()
            self.work_order[bin_] = order['item']
            work_order_list.append({'bin': bin_, 'item': order['item']})
        rospy.set_param('~work_order', work_order_list)

        self.object_data = None
        if self.is_apc2016:
            self.object_data = jsk_apc2016_common.get_object_data()

        self.bin_contents = jsk_apc2016_common.get_bin_contents(param='~bin_contents')

        self.msg = self.get_work_order_msg()

        self.pub_left = rospy.Publisher(
            '~left_hand', WorkOrderArray, queue_size=1)
        self.pub_right = rospy.Publisher(
            '~right_hand', WorkOrderArray, queue_size=1)
        rospy.Service('~update_target', UpdateTarget, self._update_target_cb)
        self.updated = False
        rospy.Timer(rospy.Duration(rospy.get_param('~duration', 1)), self._timer_cb)

    def get_sorted_work_order(self):
        """Sort work order to maximize the score"""
        bin_order = self.work_order.keys()
        if self.object_data is not None:
            if all(self.gripper in x for x in [d['graspability'].keys() for d in self.object_data]):

                def get_graspability(bin_):
                    target_object = self.work_order[bin_]
                    target_object_data = [data for data in self.object_data
                                          if data['name'] == target_object][0]
                    graspability = target_object_data['graspability'][self.gripper]
                    return graspability

                # re-sort the work_order with graspability
                bin_order = sorted(bin_order, key=get_graspability)
            else:
                jsk_logwarn('Not sorted by graspability')
                jsk_logwarn('Not all object_data have graspability key: {gripper}'
                            .format(gripper=self.gripper))
        bin_order = sorted(self.work_order.keys(),
                           key=lambda bin_: len(self.bin_contents[bin_]))
        work_order = [(bin_, self.work_order[bin_]) for bin_ in bin_order]
        return work_order

    def get_work_order_msg(self):
        work_order = self.get_sorted_work_order()
        if self.max_weight == -1:
            self.max_weight = np.inf
        msg = dict(left=WorkOrderArray(), right=WorkOrderArray())
        if self.gripper == 'gripper_v5':
            abandon_target_objects = []
            abandon_bin_objects = []
        else:
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
        if self.gripper == 'gripper_v5':
            abandon_bins = ['h', 'i']
        else:
            abandon_bins = ''
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        for bin_, target_object in work_order:
            if bin_ in abandon_bins:
                jsk_logwarn('Skipping bin {bin} because of user request'
                            .format(bin=bin_))
                continue
            if self.object_data is not None:
                target_object_data = [data for data in self.object_data
                                      if data['name'] == target_object][0]
                if target_object_data['weight'] > self.max_weight:
                    jsk_logwarn('Skipping target {obj} in {bin_}: it exceeds max weight {weight} > {max_weight}'
                                .format(obj=target_object_data['name'], bin_=bin_,
                                        weight=target_object_data['weight'], max_weight=self.max_weight))
                    continue
                if target_object_data['graspability'][self.gripper] > 3:
                    jsk_logwarn('Skipping target {obj} in {bin_}: it exceeds graspability'
                                '{grasp} > {max_grasp} with gripper {gripper}'
                                .format(obj=target_object_data['name'], bin_=bin_,
                                        grasp=target_object_data['graspability'][self.gripper],
                                        max_grasp=3,
                                        gripper=self.gripper))
                    continue
            else:
                if target_object in abandon_target_objects:
                    jsk_logwarn('Skipping target {obj} in {bin_}: it is listed as abandon target'
                                .format(obj=target_object, bin_=bin_))
                    continue
                if any(bin_object in abandon_bin_objects for bin_object in self.bin_contents[bin_]):
                    jsk_logwarn('Skipping {bin_}: this bin contains abandon objects'.format(bin_=bin_))
                    continue
            # if len(bin_contents[bin_]) > 5:  # Level3
            #     jsk_logwarn('Skipping {bin_}: this bin contains more than 5 objects'.format(bin_=bin_))
            #     continue
            order = WorkOrder(bin=bin_, object=target_object)
            if bin_ in 'abdegj':
                msg['left'].array.append(order)
            elif bin_ in 'cfhikl':
                msg['right'].array.append(order)
        return msg

    def _timer_cb(self, event):
        if self.updated:
            self.msg = self.get_work_order_msg()
            work_order_list = []
            for bin_, item in self.work_order.items():
                work_order_list.append({'bin': bin_, 'item': item})
            rospy.set_param('~work_order', work_order_list)
        self.pub_left.publish(self.msg['left'])
        self.pub_right.publish(self.msg['right'])
        self.updated = False

    def _update_target_cb(self, req):
        if not req.target:
            del self.work_order[req.bin]
        elif req.target in self.bin_contents[req.bin]:
            self.work_order[req.bin] = req.target
        else:
            rospy.logerr(
                'Unexpected target object is passed: {} for bin_contents {}'
                .format(req.target, self.bin_contents[req.bin]))
        self.updated = True
        return UpdateTargetResponse(success=True)


if __name__ == "__main__":
    rospy.init_node('work_order_server')
    work_order_server = WorkOrderServer()
    rospy.spin()
