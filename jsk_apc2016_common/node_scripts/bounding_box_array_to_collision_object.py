#!/usr/bin/env python

import time

import tf
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from jsk_recognition_msgs.msg import PolygonArray
from jsk_recognition_msgs.msg import BoundingBoxArray
import moveit_commander
import rospy
import tf2_geometry_msgs.tf2_geometry_msgs as tf_utils
from jsk_topic_tools import ConnectionBasedTransport

import numpy as np


def callback(box_arr_msg):
    for name in scene.get_known_object_names():
        if name.startswith('polygon_'):
            scene.remove_world_object(name)

    for i, box_msg in enumerate(box_arr_msg.boxes):
        pose = PoseStamped()
        pose.header.frame_id = box_arr_msg.header.frame_id
        pose.header.stamp = box_arr_msg.header.stamp
        pose.pose.position.x = box_msg.pose.position.x
        pose.pose.position.y = box_msg.pose.position.y
        pose.pose.position.z = box_msg.pose.position.z
        pose.pose.orientation.w = box_msg.pose.orientation.w
        size = (box_msg.dimensions.x, box_msg.dimensions.y, box_msg.dimensions.z)
        rospy.loginfo('Adding polygon_%04d' % i)
        scene.add_box(
            name = 'polygon_%04d' % i,
            pose = pose,
            size = size,
        )


if __name__ == '__main__':
    rospy.init_node('bounding_box_array_to_collision_object')

    scene = moveit_commander.PlanningSceneInterface()
    time.sleep(1)

    sub = rospy.Subscriber('~input', BoundingBoxArray, callback)

    rospy.spin()
