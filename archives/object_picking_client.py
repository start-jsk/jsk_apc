#!/usr/bin/env python

import os
import argparse

import rospy
import actionlib
from jsk_2015_05_baxter_apc.msg import (
    ObjectPickingAction,
    ObjectPickingGoal,
    )


def main():
    # cmdline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--limb', choices=('l', 'r'),
        help='left or right arm', required=True)
    parser.add_argument('-r', '--to-release', action='store_true',
        help='to pick or release object')
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("object_picking_client")
    client = actionlib.SimpleActionClient("object_picking",
                                          ObjectPickingAction)
    print("{} wait_for_server".format(os.getpid()))
    client.wait_for_server()

    goal = ObjectPickingGoal()
    goal.limb = 'left' if args.limb == 'l' else 'right'
    goal.state = not args.to_release
    print("Requesting ...")

    client.send_goal(goal)

    print("{} wait_for_result".format(os.getpid()))
    client.wait_for_result(rospy.Duration.from_sec(10.0))

    result = client.get_result()
    if result:
        print(result.sequence)
    else:
        print("get result None.")


if __name__ == "__main__":
    main()

