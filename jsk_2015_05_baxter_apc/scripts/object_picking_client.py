#!/usr/bin/env python

import os

import rospy
import actionlib

from jsk_2014_picking_challenge.msg import *

def main():
    rospy.init_node("object_picking_client")
    client = actionlib.SimpleActionClient("object_picking", ObjectPickingAction)
    print("{} wait_for_server".format(os.getpid()))
    client.wait_for_server()

    goal = ObjectPickingGoal()
    goal.limb = 'larm'
    goal.state = False
    print("Requesting ...")

    client.send_goal(goal)

    print("{} wait_for_result".format(os.getpid()))
    client.wait_for_server(rospy.Duration.from_sec(10.0))

    result = client.get_result()
    print result

if __name__ == "__main__":
    main()
