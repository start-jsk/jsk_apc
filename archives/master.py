#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import datetime

import numpy as np

import rospy
import actionlib

from jsk_rviz_plugins.msg import OverlayText
from jsk_2015_05_baxter_apc.msg import (
    order_list,
    bins_content,
    MoveArm2TargetBinAction,
    MoveArm2TargetBinGoal,
    ObjectPickingAction,
    ObjectPickingGoal,
    )
from jsk_2015_05_baxter_apc.srv import (
    MoveArm,
    ReleaseItem,
    ObjectMatch
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

        self.pub_rvizmsg = rospy.Publisher('/semi/master_status', OverlayText,
                                           queue_size=1)
        self.rvizmsg = []
        self.sub_bin_contents = rospy.Subscriber(
            '/semi/bin_contents', bins_content, self.cb_bin_contents)
        self.sub_orders = rospy.Subscriber(
            '/semi/order_list', order_list, self.cb_orders)
        rospy.wait_for_message('/semi/bin_contents', bins_content)
        rospy.wait_for_message('/semi/order_list', order_list)
        self.pub_rvizmsg.publish(text='Initialized')

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

    def move_for_verification(self):
        client = rospy.ServiceProxy('/move_for_verification', MoveArm)
        res = client(limb=self.use_limb)
        return res

    def object_verification(self):
        """Verify item if it's intended one."""
        bin_name = self.target[0]
        target_object = self.target[1]
        if len(self.bin_contents[bin_name]) == 1:
            self.mode = 'place_item'
            return True
        res = self.move_for_verification()
        matcher_client = rospy.ServiceProxy('/bof_matcher', ObjectMatch)
        res = matcher_client(objects=self.bin_contents[bin_name])
        if target_object == np.argmax(res.probabilities):
            self.mode = 'place_item'
            return True
        else:
            self.mode = 'move2target'
            return False

    def place_item(self):
        """Place item into order bin."""
        # target_obj = self.target[1]
        limb = self.use_limb
        lorr = 'l' if limb == 'left' else 'r'
        client = rospy.ServiceProxy('/semi/{}arm_put_orderbin'.format(lorr),
            ReleaseItem)
        client.wait_for_service()
        res = client()
        self.target = None
        self.mode = 'move2target'
        rospy.loginfo('Finished: place_item')
        return res.succeeded

    def publish_rvizmsg(self, text):
        self.rvizmsg.append(datetime.datetime.now().strftime("%H:%M:%S ")+text.replace('\n', '\n          '))
        # if text exceeds 5 line, remove first
        if len(self.rvizmsg) > 3:
            self.rvizmsg.pop(0)
        self.pub_rvizmsg.publish(text='\n'.join(self.rvizmsg))

    def main(self):
        bin_contents = self.bin_contents
        orders = self.orders

        go_bin_orders = list('abcdefghijkl')  # bin_names for orders
        while len(bin_contents) > 0:
            rospy.loginfo('Current mode is {}'.format(self.mode))
            # decide target
            if not self.target:
                bin_name = go_bin_orders.pop(0)
                order_item = orders[bin_name]
                self.target = (bin_name, order_item)
            target_text = 'Target bin:{}, item:{}'.format(
                self.target[0], self.target[1])
            self.pub_rvizmsg.publish(text=target_text)
            # decide action
            if self.mode == 'move2target':
                self.publish_rvizmsg(text='Move arm to target bin\n'
                                         + target_text)
                is_success = self.move2target()
                self.publish_rvizmsg(text='Move arm success?: {}'.format(
                                 is_success))
            elif self.mode == 'grasp_ctrl':
                self.publish_rvizmsg(text='Insert arm to target bin\n'
                                         + target_text)
                self.grasp_ctrl(to_grasp=False)
                is_success = self.grasp_ctrl(to_grasp=True)
                self.publish_rvizmsg(text='Grasp success?: {}'.format(
                                         is_success))
            elif self.mode == 'object_verification':
                self.publish_rvizmsg(text='Object verification\n'
                                         + target_text)
                is_correct = self.object_verification()
                self.publish_rvizmsg(text='Correct item?: {}'.format(
                                         is_correct))
            elif self.mode == 'place_item':
                self.publish_rvizmsg(text='Place item to order bin\n'+target_text)
                is_success = self.place_item()
                self.publish_rvizmsg(
                    text='Placing item success?: {}'.format(is_success))
                

if __name__ == '__main__':
    m = Master()
    m.main()
