#!/usr/bin/env python
#
import os

import rospy
import actionlib
# from actionlib_tutorials.msg import FibonacciAction, FibonacciGoal
from jsk_2014_picking_challenge.msg import (
    MoveArm2TargetBinAction,
    MoveArm2TargetBinGoal,
)


def main():
    rospy.init_node('move_arm2target_bin_client')
    client = actionlib.SimpleActionClient('move_arm2target_bin', MoveArm2TargetBinAction)
    print("{0} wait_for_server".format(os.getpid()))
    client.wait_for_server()

    goal = MoveArm2TargetBinGoal()
    goal.order = 'a'
    print "Requesting fibonacci {0}".format(goal.order)

    client.send_goal(goal)

    print("{0} wait_for_result".format(os.getpid()))
    client.wait_for_result(rospy.Duration.from_sec(10.0))

    result = client.get_result()
    print("Resulting fibonacci {0}".format(result.sequence))


if __name__ == '__main__':
    main()

