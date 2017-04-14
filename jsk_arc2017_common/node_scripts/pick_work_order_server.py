#!/usr/bin/env python

from jsk_arc2017_common.msg import WorkOrder
from jsk_arc2017_common.msg import WorkOrderArray
import json
import rospy


class WorkOrderServer(object):

    def __init__(self):
        self.json_file = rospy.get_param('~json', None)
        self.rate = rospy.get_param('~rate', 1.0)
        if self.json_file is None:
            rospy.logerr('must set json file path to ~json')
            return
        with open(self.json_file) as f:
            data = json.load(f)
        self.bin_contents = data['bin_contents']
        self.target_items = data['target_items']
        larm_box_list = ['box_A']
        rarm_box_list = ['box_B', 'box_C']
        self.larm_msg = self._generate_msg(larm_box_list)
        self.rarm_msg = self._generate_msg(rarm_box_list)
        self.larm_pub = rospy.Publisher(
            '~left_hand', WorkOrderArray, queue_size=1)
        self.rarm_pub = rospy.Publisher(
            '~right_hand', WorkOrderArray, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.rate), self._publish_msg)

    def _generate_msg(self, box_list):
        orders = []
        for box in box_list:
            for target_item in self.target_items[box]:
                order = WorkOrder()
                order.bin = self._get_target_bin(target_item)
                order.item = target_item
                order.box = box
                orders.append(order)
        msg = WorkOrderArray()
        msg.orders = orders
        return msg

    def _get_target_bin(self, target_item):
        for key, values in self.bin_contents.items():
            if target_item in values:
                target_bin = key
                break
        return target_bin

    def _publish_msg(self, event):
        self.larm_pub.publish(self.larm_msg)
        self.rarm_pub.publish(self.rarm_msg)

if __name__ == "__main__":
    rospy.init_node('work_order_server')
    work_order_server = WorkOrderServer()
    rospy.spin()
