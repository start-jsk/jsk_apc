#!/usr/bin/env python

import os

import rospy
import actionlib

from jsk_2014_picking_challenge.msg import *

def main():
    rospy.init_node("object_picking_test")

    client_names = ["move_arm2target_bin", "object_picking", "move_arm2target_bin"]
    for client_name in client_names:
        print("client_name is {}".format(client_name))
        client = actionlib.SimpleActionClient(client_name, MoveArm2TargetBinAction)
        print("{} wait_for_server".format(os.getpid()))
        client.wait_for_server()

        goal = MoveArm2TargetBinGoal()
        goal.order = 'a'

        client.send_goal(goal)

        print("{} wait_for_result".format(os.getpid()))
        client.wait_for_result(rospy.Duration.from_sec(10.0))

        result = client.get_result()
        # print("Resulting fibonacci {}".format(result.sequence))
        print(result)

if __name__ == "__main__":
    main()
