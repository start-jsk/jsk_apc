#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import rospy
from std_msgs.msg import Bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['start', 'stop'])
    limbs = ['left', 'right']
    parser.add_argument('limb', type=str, choices=limbs, nargs='?')
    args = parser.parse_args()

    action = args.action
    limbs = ['left', 'right'] if args.limb is None else [args.limb]

    rospy.init_node('control_vacuum_gripper')

    pubs = []
    for limb in limbs:
        pub = rospy.Publisher(
            '/vacuum_gripper/limb/{}'.format(limb), Bool, queue_size=1)
        pubs.append(pub)

    # this sleep is necessary to register publisher in actual
    rospy.sleep(1)

    for limb, pub in zip(limbs, pubs):
        print('{action}-ing {limb} hand vacuum gripper'
              .format(action=action, limb=limb))
        pub.publish(Bool(data=action == 'start'))


if __name__ == '__main__':
    main()
