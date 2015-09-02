#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import rospy
from std_msgs.msg import Bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hand', help='right or left')
    args = parser.parse_args()

    rospy.init_node('toggle_vacuum', anonymous=True)
    pub = rospy.Publisher('/vacuum_gripper/limb/{0}'.format(args.hand),
                          Bool, queue_size=1)

    state = False
    while True:
        input = raw_input('Please enter any key (q or quit to exit): ')
        if input in ['q', 'quit']:
            break
        state = not state
        print('Vacuum state: {0}'.format(state))
        pub.publish(state)


if __name__ == '__main__':
    main()