#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import argparse

import roslib; roslib.load_manifest("jsk_2014_picking_challenge")
import rospy
from std_msgs.msg import String
from jsk_2014_picking_challenge.msg import order_list
from jsk_2014_picking_challenge.msg import one_order

def main():
    """Baxter read jsonfile."""
    arg_fmt = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-f', '--file', required=True,
        help='select json file.'
    )
    args = parser.parse_args(rospy.myargv()[1:])


    f = open(args.file, 'r')
    jsonData = json.load(f)

    bin_contents = jsonData['bin_contents']
    for bin in bin_contents:
        s = ""
        for content in bin_contents[bin]:
            s += content + ":"
        s = s[:len(s)-1]
        print("{0}:{1}".format(bin, s))

    order_data = order_list()
    work_orders = jsonData['work_order']
    for order in work_orders:
        data = one_order()
        data.bin = order['bin']
        data.item = order['item']
        order_data.order_list.append(data)

    rospy.loginfo(order_data)
    rospy.init_node('read_json_data')
    pub = rospy.Publisher('read_json_data', order_list)

    while not rospy.is_shutdown():
        rospy.loginfo(order_data)
        pub.publish(order_data)
        rospy.sleep(1.0)


if __name__ == "__main__":
    main()
