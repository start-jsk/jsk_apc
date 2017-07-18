#!/usr/bin/env python

import jsk_arc2017_common
from jsk_arc2017_common.msg import WorkOrder
from jsk_arc2017_common.msg import WorkOrderArray
import json
import operator
import os.path as osp
import random
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
        box_path = osp.join(json_dir, 'box_sizes.json')
        with open(box_path) as box_f:
            boxes = json.load(box_f)['boxes']

        self.item_location = {}
        for bin_ in bins:
            bin_id = bin_['bin_id']
            for item_name in bin_['contents']:
                self.item_location[item_name] = bin_id

        box_sizes = {}
        for box in boxes:
            size_id = box['size_id']
            box_sizes[size_id] = reduce(operator.mul, box['dimensions'])
        size_ids = [order['size_id'] for order in orders]
        sorted_size_ids = sorted(size_ids, key=lambda x: box_sizes[x])

        self.cardboard_ids = {}
        for i, size_id in enumerate(sorted_size_ids):
            self.cardboard_ids[size_id] = 'ABC'[i]

        publish_orders = self._generate_publish_orders(orders)

        object_weights = jsk_arc2017_common.get_object_weights()
        left_sorted_orders = sorted(
            publish_orders['left_hand'],
            key=lambda order: object_weights[order['item']])
        right_sorted_orders = sorted(
            publish_orders['right_hand'],
            key=lambda order: object_weights[order['item']])

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
                    hand = 'left_hand'
                elif bin_ == 'C':
                    hand = 'right_hand'
                else:  # bin_ == 'B'
                    right_length = len(publish_orders['right_hand'])
                    left_length = len(publish_orders['left_hand'])
                    if right_length > left_length:
                        hand = 'left_hand'
                    elif left_length > right_length:
                        hand = 'right_hand'
                    else:
                        hand = random.choice(['left_hand', 'right_hand'])
                publish_orders[hand].append(order)
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
