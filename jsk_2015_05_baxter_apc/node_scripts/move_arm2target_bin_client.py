#!/usr/bin/env python
#
import os
import argparse

import rospy
import actionlib
from jsk_2014_picking_challenge.msg import (
    MoveArm2TargetBinAction,
    MoveArm2TargetBinGoal,
)


def main():
    # cmdline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('order', help='bin name for arm to move')
    args = parser.parse_args(rospy.myargv()[1:])

    # initialize
    rospy.init_node('move_arm2target_bin_client')
    client = actionlib.SimpleActionClient('move_arm2target_bin', MoveArm2TargetBinAction)

    # wait for server to boot
    print("{0} wait_for_server".format(os.getpid()))
    client.wait_for_server()

    # request goal for arm to move
    goal = MoveArm2TargetBinGoal()
    goal.order = args.order
    print "Requesting move for bin {0}".format(goal.order)
    client.send_goal(goal)

    # wait for result
    print("{0} wait_for_result".format(os.getpid()))
    client.wait_for_result(rospy.Duration.from_sec(20.0))

    # get result
    result = client.get_result()
    if result:
        print("Resulting move for bin {0}".format(result.sequence))


if __name__ == '__main__':
    main()

