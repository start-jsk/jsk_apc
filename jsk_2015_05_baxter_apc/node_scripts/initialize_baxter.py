#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import subprocess
from multiprocessing import Pool

import rospy

import baxter_interface
from baxter_interface import CHECK_VERSION

from rosgraph_msgs.msg import Clock
from baxter_core_msgs.msg import AssemblyState

import rospkg


def main():
    rospy.init_node('initialize_baxter')

    rospy.loginfo('waiting for /clock & /robot/state')
    rospy.wait_for_message('/clock', Clock)
    rospy.wait_for_message('/robot/state', AssemblyState)
    rospy.loginfo('found /clock & /robot/state')

    # enable robot
    baxter_interface.RobotEnable(CHECK_VERSION).enable()

    # joint action server
    rospack = rospkg.RosPack()
    pool = Pool(processes=2)
    path = rospack.get_path('baxter_interface')
    commands = [
        ('rosrun', 'baxter_interface', 'joint_trajectory_action_server.py'),
        ('rosrun', 'baxter_interface', 'head_action_server.py'),
        ]
    pool.map(subprocess.call, commands)


if __name__ == '__main__':
    main()
