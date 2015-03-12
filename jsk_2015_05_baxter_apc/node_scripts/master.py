#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import sys
import numpy as np
from collections import OrderedDict

import rospy
import actionlib

from jsk_2014_picking_challenge.msg import (
    order_list,
    bins_content,
    MoveArm2TargetBinAction,
    MoveArm2TargetBinGoal,
    ObjectPickingAction,
    ObjectPickingGoal,
    )
from jsk_2014_picking_challenge.srv import (
    ReleaseItem,
    ObjectVerification,
    )


class Master(object):
    """Master program for Amazon Picking Challenge. The strategy is belows::
        * Move arm in the front of target bin.
        * Pick item from the bin randomly.
        * Show it to camera and verify if it's intended one.
        * Place it to order bin.
    """
    def __init__(self):
        rospy.init_node('jsk_2014_semi_master')

        self.use_limb = 'right'
        self.mode = 'move2target'
        self.target = None

        self.sub_bin_contents = rospy.Subscriber(
            '/semi/bin_contents', bins_content, self.cb_bin_contents)
        self.sub_orders = rospy.Subscriber(
            '/semi/order_list', order_list, self.cb_orders)
        rospy.wait_for_message('/semi/bin_contents', bins_content)
        rospy.wait_for_message('/semi/order_list', order_list)

    def cb_bin_contents(self, msg):
        """Get bin contents information from topic."""
        bins = msg.bins
        bin_contents = {bin_.bin[-1].lower(): bin_.items for bin_ in bins}
        self.bin_contents = bin_contents
        self.sub_bin_contents.unregister()

    def cb_orders(self, msg):
        """Get order information from topic."""
        orders = msg.order_list
        orders = {order.bin[-1].lower(): order.item for order in orders}
        self.orders = orders
        self.sub_orders.unregister()

    def move2target(self):
        """Move arm in the front of target bin."""
        bin_name = self.target[0]
        limb = self.use_limb
        if bin_name not in 'abcdefghijkl':
            rospy.logerr('Invalid target bin: {}'.format(bin_name))
        rospy.loginfo('Moving arm to {}'.format(bin_name))

        client = actionlib.SimpleActionClient('move_arm2target_bin',
                                              MoveArm2TargetBinAction)
        rospy.loginfo('Waiting for service: move_arm2target_bin')
        client.wait_for_server()
        rospy.loginfo('Found service: move_arm2target_bin')
        # send goal
        goal = MoveArm2TargetBinGoal(limb=limb, order=bin_name)
        client.send_goal(goal)
        # get result
        for trial in xrange(3):
            client.wait_for_result(rospy.Duration.from_sec(30))
            result = client.get_result()
            state = client.get_state()
            if result and state == 3:
                rospy.loginfo("Move result: {}".format(result.sequence))
                self.mode = 'grasp_ctrl'
                return True
        if result:
            rospy.loginfo(result.sequence)
        rospy.logwarn('Failed motion: move2target')
        self.target = None  # abandon current target
        self.mode = 'move2target'
        return False

    def grasp_ctrl(self, to_grasp):
        """Pick item from target bin randomly."""
        limb = self.use_limb
        target = self.target
        if to_grasp:
            rospy.loginfo('Getting {item} in {bin}'.format(
                bin=target[0], item=target[1]))
        else:
            rospy.loginfo('Returning item to {bin}'.format(bin=target[0]))
        client = actionlib.SimpleActionClient("object_picking",
                                              ObjectPickingAction)
        rospy.loginfo('Waiting for service: object_picking')
        client.wait_for_server()
        rospy.loginfo('Found service: object_picking')
        # send result
        goal = ObjectPickingGoal(limb=limb, state=to_grasp)
        client.send_goal(goal)
        # get result
        for trial in xrange(3):
            client.wait_for_result(rospy.Duration.from_sec(30))
            result = client.get_result()
            state = client.get_state()
            if result and state == 3:
                rospy.loginfo('Grasp result: {}'.format(result.sequence))
                if to_grasp:
                    self.mode = 'object_verification'
                else:
                    self.mode = 'grasp_ctrl'
                return True
        if result:
            rospy.loginfo(result.sequence)
        rospy.logwarn('Failed motion: grasp_ctrl')
        self.target = None  # abandon current target
        self.mode = 'move2target'
        return False

    def object_verification(self):
        """Verify item if it's intended one."""
        bin_name = self.target[0]
        target_object = self.target[1]
        if len(self.bin_contents[bin_name]) == 1:
            self.mode = 'place_item'
            return True
        limb = self.use_limb
        lorr = 'l' if limb == 'left' else 'r'
        client = rospy.ServiceProxy(
            '/semi/{}arm_move_for_verification'.format(lorr),
            ObjectVerification)
        res = client(objects=self.bin_contents[bin_name],
                     target_object=target_object)
        rospy.loginfo('Waiting for server: move_for_verification')
        client.wait_for_service()
        rospy.loginfo('Found: move_for_verification')
        if res.succeeded:
            self.mode = 'place_item'
            return True
        else:
            rospy.logwarn('Failed motion: object_verification')
            self.mode = 'move2target'
            return False

    def place_item(self):
        """Place item into order bin."""
        limb = self.use_limb
        lorr = 'l' if limb == 'left' else 'r'
        client = rospy.ServiceProxy('/semi/{}arm_put_orderbin'.format(lorr),
            ReleaseItem)
        client.wait_for_service()
        res = client()
        if res.succeeded:
            self.mode = 'move2target'
            return True
        else:
            rospy.logwarn('Failed to place item: {}'.format(self.target[1]))
            self.target = None  # abandon current target
            self.mode = 'move2target'
            return False

    def main(self):
        bin_contents = self.bin_contents
        orders = self.orders

        bin_contents= sorted(bin_contents.items(), key=lambda x: len(x[1]))
        while len(bin_contents) > 0:
            # decide target
            if not self.target:
                bin_name = bin_contents.pop(0)[0]
                order_item = orders[bin_name]
                self.target = (bin_name, order_item)
            # decide action
            if self.mode == 'move2target':
                self.move2target()
            elif self.mode == 'grasp_ctrl':
                self.grasp_ctrl(to_grasp=False)
                self.move2target()
                self.grasp_ctrl(to_grasp=True)
            elif self.mode == 'object_verification':
                self.object_verification()
            elif self.mode == 'place_item':
                self.place_item()


if __name__ == '__main__':
    m = Master()
    m.main()

