#!/usr/bin/env python
# -*- coding: utf-8 -*-

from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import Pose
import rospy

#Segmmentation Fault
import atexit, os
atexit.register(lambda : os._exit(0))

group = MoveGroupCommander("right_arm")

rospy.init_node("fetch_moveit")

target_pose = Pose()
target_pose.position.x = -0.3
target_pose.position.z = 0.7
target_pose.orientation.w = 1

group.set_pose_target(target_pose)

group.go()
