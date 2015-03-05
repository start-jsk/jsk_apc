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
    )
from jsk_2014_picking_challenge.srv import ReleaseItem


class Master(object):
    """Master program for Amazon Picking Challenge. The strategy is belows::
        * Move arm in the front of target bin.
        * Pick item from the bin randomly.
        * Show it to camera and verify if it's intended one.
        * Place it to order bin.
    """
    def __init__(self):
        rospy.init_node('jsk_2014_semi_master')

        self.use_limb = 'left'
        self.mode = 'mv2target'
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

    def client_mv2target(self):
        """Move arm in the front of target bin."""
        bin_name = self.target[0]
        limb = self.use_limb
        assert bin_name in 'abcdefghijkl', 'invalid target bin: {}'.format(bin_name)
        assert limb in ['left', 'right'], 'invalid limb: {}'.format(limb)
        rospy.loginfo('Moving to {}'.format(bin_name))
        #
        mv2target = actionlib.SimpleActionClient(
            'move_arm2target_bin', MoveArm2TargetBinAction)
        mv2target.wait_for_server()
        # send goal
        goal = MoveArm2TargetBinGoal()
        goal.limb = limb
        goal.order = bin_name
        mv2target.send_goal(goal)
        # get result
        for trial in xrange(3):
            mv2target.wait_for_result(rospy.Duration.from_sec(10.0))
            result = mv2target.get_result()
            state = mv2target.get_state()
            if result and state == 3:
                rospy.loginfo("Move result for bin {}".format(result.sequence))
                self.mode = 'grasp_ctrl'
                return True
        self.target = None  # abandon current target
        self.mode = 'mv2target'
        return False

    def client_grasp_ctrl(self, limb, state):
        """Pick item from target bin randomly."""
        target = self.target
        rospy.loginfo('Getting {item} in {bin}'.format(
            bin=target[0], item=target[1]))
        #     return False
        client = actionlib.SimpleActionClient("object_picking", ObjectPickingAction)
        print("{} wait_for_server".format(os.getpid()))
        client.wait_for_server()

        goal = ObjectPickingGoal()
        goal.limb = limb
        goal.state = state

        client.send_goal(goal)

        print("{} wait_for_result".format(os.getpid()))
        client.wait_for_result(rospy.Duration.from_sec(10.0))

        result = client.get_result
        if result:
            print("{}".format(result.sequence))
        else:
            print("get result None.")

    def client_item_verification(self, item_name):
        """Verify item if it's intended one."""
        # item_verification = rospy.ServiceProxy(
        #     'item_verification', ReleaseItem)
        # item_verification.wait_for_service()
        # res = item_verification()
        # if res.succeeded:
        #     self.mode = 'place_item'
        #     return True
        # else:
        #     self.mode = 'mv2target'
        #     return False
        raise NotImplementedError("waiting Tnoriaki implementation")

    def client_place_item(self):
        """Place item into order bin."""
        limb = self.use_limb
        assert limb in ['left', 'right']
        lorr = 'l' if limb == 'left' else 'r'
        place_item = rospy.ServiceProxy('/semi/{}arm_put_orderbin'.format(lorr), ReleaseItem)
        place_item.wait_for_service()
        res = place_item()
        if res.succeeded is False:
            rospy.logwarn('Failed to place item')
            return False
        self.target = None  # abandon current target
        self.mode = 'mv2target'
        return True

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
            return  # because some methods is not implemented
            # decide action
            if self.mode == 'move2target':
                self.client_mv2target()
            elif self.mode == 'grasp_ctrl':
                self.client_grasp_ctrl(self.use_limb, True)  # NotImplemented
                pass
            elif self.mode == 'verify_item':
                # self.client_item_verification(target)  # NotImplemented
                pass
            elif self.mode == 'place_item':
                self.client_place_item()


if __name__ == '__main__':
    m = Master()
    m.main()
