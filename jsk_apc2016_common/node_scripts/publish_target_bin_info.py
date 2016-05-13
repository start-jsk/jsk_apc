#!/usr/bin/env python

import rospy
from jsk_apc2016_common.msg import BinInfo, BinInfoArray


def bin_info_array_to_dict(bin_info_array):
    bin_info_dict = {}
    for bin_ in bin_info_array.array:
        bin_info_dict[bin_.name] = bin_
    return bin_info_dict


def main():
    bin_info_array = rospy.wait_for_message(
            "~input/bin_info_array", BinInfoArray, timeout=50)
    target_bin_info_pub = rospy.Publisher('~target_bin_info', BinInfo, queue_size=5)

    bin_info_dict = bin_info_array_to_dict(bin_info_array)

    rate = rospy.Rate(rospy.get_param('rate', 10))
    while not rospy.is_shutdown():
        target_bin_name = rospy.get_param('~target_bin_name')
        if target_bin_name not in 'abcdefghijkl' or target_bin_name == '':
            rate.sleep()
            continue
        bin_info_dict[target_bin_name].header.seq = (
                bin_info_dict[target_bin_name].header.seq + 1)
        target_bin_info = bin_info_dict[target_bin_name]
        target_bin_info.header.stamp = rospy.Time.now()
        target_bin_info_pub.publish(target_bin_info)
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('publish_target_bin_info')
    main()
