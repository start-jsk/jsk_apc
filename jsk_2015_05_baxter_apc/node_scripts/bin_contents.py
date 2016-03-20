#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json

import rospy
from jsk_2015_05_baxter_apc.msg import BinContents, BinContentsArray
from jsk_recognition_msgs.msg import Int32Stamped


def get_bin_contents(json_file):
    with open(json_file, 'r') as f:
        bin_contents = json.load(f)['bin_contents']
    for bin_, objects in bin_contents.items():
        bin_ = bin_.split('_')[1].lower()  # bin_A -> a
        yield (bin_, objects)


def main():
    json_file = rospy.get_param('~json', None)
    if json_file is None:
        rospy.logerr('must set json file path to ~json')
        return
    bin_contents = get_bin_contents(json_file=json_file)
    bin_contents = list(bin_contents)

    pubs_n_obj = []
    msg = BinContentsArray()
    for bin_, objects in bin_contents:
        msg.array.append(BinContents(bin=bin_, objects=objects))
        pub = rospy.Publisher('~bin_{}_n_object'.format(bin_),
                              Int32Stamped, queue_size=1)
        pubs_n_obj.append(pub)

    pub_contents = rospy.Publisher('~', BinContentsArray, queue_size=1)
    rate = rospy.Rate(rospy.get_param('rate', 1))
    while not rospy.is_shutdown():
        pub_contents.publish(msg)
        for pub, (_, objects) in zip(pubs_n_obj, bin_contents):
            msg_n_obj = Int32Stamped()
            msg_n_obj.header.stamp = rospy.Time.now()
            msg_n_obj.data = len(objects)
            pub.publish(msg_n_obj)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('bin_contents')
    main()
