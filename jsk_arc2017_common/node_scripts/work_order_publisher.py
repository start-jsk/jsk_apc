#!/usr/bin/env python

import jsk_arc2017_common
from jsk_arc2017_common.msg import WorkOrder
from jsk_arc2017_common.msg import WorkOrderArray
import json
import os.path as osp
import rospy


class WorkOrderPublisher(object):

    abandon_items = [
        'measuring_spoons',
        'mesh_cup'
        ]

    def __init__(self):
        json_dir = rospy.get_param('~json_dir', None)
        self.rate = rospy.get_param('~rate', 1.0)
        if json_dir is None:
            rospy.logerr('must set json dir path to ~json_dir')
            return
        location_path = osp.join(json_dir, 'item_location_file.json')
        with open(location_path) as location_f:
            bins = json.load(location_f)['bins']
        order_path = osp.join(json_dir, 'order_file.json')
        with open(order_path) as order_f:
            orders = json.load(order_f)['orders']

        self.item_location = {}
        for bin_ in bins:
            bin_id = bin_['bin_id']
            for item_name in bin_['contents']:
                self.item_location[item_name] = bin_id

        self.cardboard_ids = {}
        for order in orders:
            size_id = order['size_id']
            if len(order['contents']) == 2:
                self.cardboard_ids[size_id] = 'A'
            elif len(order['contents']) == 3:
                self.cardboard_ids[size_id] = 'B'
            else:  # len(order['contents']) == 5
                self.cardboard_ids[size_id] = 'C'

        publish_orders = self._generate_publish_orders(orders)

        # first: sort by object weight
        object_weights = jsk_arc2017_common.get_object_weights()
        left_sorted_orders = sorted(
            publish_orders['left_hand'],
            key=lambda order: object_weights[order['item']])
        right_sorted_orders = sorted(
            publish_orders['right_hand'],
            key=lambda order: object_weights[order['item']])

        # second: sort by object graspability
        graspability = jsk_arc2017_common.get_object_graspability()
        left_sorted_orders = sorted(
            left_sorted_orders,
            key=lambda order: graspability[order['item']]['suction'])
        right_sorted_orders = sorted(
            right_sorted_orders,
            key=lambda order: graspability[order['item']]['suction'])

        self.larm_msg = self._generate_msg(left_sorted_orders)
        self.rarm_msg = self._generate_msg(right_sorted_orders)
        self.larm_pub = rospy.Publisher(
            '~left_hand', WorkOrderArray, queue_size=1)
        self.rarm_pub = rospy.Publisher(
            '~right_hand', WorkOrderArray, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.rate), self._publish_msg)

    def _generate_publish_orders(self, orders):
        publish_orders = {
            'left_hand': [],
            'right_hand': []
        }

        for order in orders:
            size_id = order['size_id']
            for item in order['contents']:
                bin_ = self.item_location[item]
                order = {
                    'item': item,
                    'bin': bin_,
                    'box': self.cardboard_ids[size_id]
                }
                if bin_ == 'A':
                    publish_orders['left_hand'].append(order)
                elif bin_ == 'C':
                    publish_orders['right_hand'].append(order)
                else:  # bin_ == 'B'
                    publish_orders['left_hand'].append(order)
                    publish_orders['right_hand'].append(order)

        return publish_orders

    def _generate_msg(self, orders):
        order_msgs = []
        for order in orders:
            target_item = order['item']
            if target_item in self.abandon_items:
                continue
            order_msg = WorkOrder()
            order_msg.bin = order['bin']
            order_msg.item = target_item
            order_msg.box = order['box']
            order_msgs.append(order_msg)
        order_array_msg = WorkOrderArray()
        order_array_msg.orders = order_msgs
        return order_array_msg

    def _publish_msg(self, event):
        self.larm_pub.publish(self.larm_msg)
        self.rarm_pub.publish(self.rarm_msg)

if __name__ == '__main__':
    rospy.init_node('work_order_publisher')
    work_order_publisher = WorkOrderPublisher()
    rospy.spin()
