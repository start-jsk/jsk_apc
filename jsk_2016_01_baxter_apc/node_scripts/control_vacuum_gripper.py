#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import rospy
from std_msgs.msg import Bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--left', action='store_true',
                        help='Control left gripper')
    parser.add_argument('-r', '--right', action='store_true',
                        help='Control right gripper')
    parser.add_argument('-t', '--start', action='store_true',
                        help='Start vacuum gripper')
    parser.add_argument('-p', '--stop', action='store_true',
                        help='Stop vacuum gripper')
    args = parser.parse_args()

    if args.start and not args.stop:
        action = 'start'
    elif args.stop:
        action = 'stop'
    else:
        print('Please specify one of start or stop action.')
        parser.print_help()
        quit(1)
    if args.left and not args.right:
        limbs = ['left']
    elif args.right:
        limbs = ['right']
    else:
        limbs = ['left', 'right']

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
