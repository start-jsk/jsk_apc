#!/usr/bin/env python

import os

import rospy
import actionlib

from jsk_2014_picking_challenge.msg import *

def main():
    rospy.init_node("object_picking_client")
    # client = actionlib.SimpleActionClient("object_picking", MoveArm2TargetBinAction)
    client = actionlib.SimpleActionClient("object_picking", ObjectPickingAction)
    print("{} wait_for_server".format(os.getpid()))
    client.wait_for_server()

    goal = MoveArm2TargetBinGoal()
    goal.limb = 'left'
    print("Requesting ...")

    client.send_goal(goal)

    print("{} wait_for_result".format(os.getpid()))
    client.wait_for_server(rospy.Duration.from_sec(10.0))

    result = client.get_result()
    print(result)
    # print("Resulting {}".format(result.sequence))

if __name__ == "__main__":
    main()
