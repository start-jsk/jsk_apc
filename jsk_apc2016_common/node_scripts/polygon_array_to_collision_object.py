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
import moveit_commander
import rospy
import tf2_geometry_msgs.tf2_geometry_msgs as tf_utils

import numpy as np

flag = 1

def callback_flag(msg):
    global flag
    if msg.data == 'stop':
        flag = 0
        rospy.logerr('A flag = %d', flag)
    if msg.data == 'start!':
        flag = 1

def callback(ply_arr_msg):
    global flag
    if flag == 1:
        rospy.logerr('flag = %d', flag)
        for name in scene.get_known_object_names():
            if name.startswith('polygon_'):
                scene.remove_world_object(name)

        dst_frame = ply_arr_msg.header.frame_id
        stamp = ply_arr_msg.header.stamp
        try:
            listener.waitForTransform(
                src_frame,
                dst_frame,
                stamp,
                timeout=rospy.Duration(1),
            )
        except Exception, e:
            rospy.logerr(e)
            return
        dst_pose = listener.lookupTransform(src_frame, dst_frame, stamp)
        transform = TransformStamped()
        transform.header.frame_id = src_frame
        transform.header.stamp = stamp
        transform.child_frame_id = dst_frame
        transform.transform.translation = Vector3(*dst_pose[0])
        transform.transform.rotation = Quaternion(*dst_pose[1])

        for ply_msg in ply_arr_msg.polygons:
            # polygon -> points
            points = np.zeros((len(ply_msg.polygon.points), 3), dtype=np.float64)
            for i, pt in enumerate(ply_msg.polygon.points):
                pt_stamped = PointStamped(header=ply_arr_msg.header, point=pt)
                pt = tf_utils.do_transform_point(pt_stamped, transform).point
                points[i, 0] = pt.x
                points[i, 1] = pt.y
                points[i, 2] = pt.z

            # polygon center
            pose = PoseStamped()
            pose.header.frame_id = src_frame
            pose.header.stamp = stamp
            x_max, y_max, z_max = np.max(points, axis=0)
            x_min, y_min, z_min = np.min(points, axis=0)
            x = (x_max + x_min) / 2.0
            y = (y_max + y_min) / 2.0
            z = z_max / 2.0
            ###z = (z_max + z_min) / 2.0
            ###x, y, z = np.mean(points, axis=0)
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1
            
            # fit polygon with box
            ###size = (x_max - x_min, y_max - y_min, z_max - z_min)
            size = (x_max - x_min, y_max - y_min, z_max)

            rospy.loginfo('Adding polygon_%04d' % i)
            scene.add_box(
                name='polygon_%04d' % i,
                pose=pose,
                size=size,
            )


if __name__ == '__main__':
    rospy.init_node('polygon_array_to_collision_object')

    scene = moveit_commander.PlanningSceneInterface()
    time.sleep(1)

    listener = tf.listener.TransformListener()

    src_frame = rospy.get_param('~fixed_frame', 'base_link')

    rospy.Subscriber('stop_moveit', String, callback_flag)

    sub = rospy.Subscriber('~input', PolygonArray, callback)

    rospy.spin()
