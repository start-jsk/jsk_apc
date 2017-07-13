#!/usr/bin/env python

import os
import os.path as osp

import numpy as np
import pandas

from geometry_msgs.msg import PoseStamped
import rospy
import tf


class TestHandEyeCoordination(object):

    def __init__(self):
        self.out_file = rospy.get_param('~out_file')
        if not osp.exists(osp.dirname(self.out_file)):
            os.makedirs(osp.dirname(self.out_file))

        self.tf_listener = tf.listener.TransformListener()
        # DEBUG
        # self.pub_debug = rospy.Publisher('~debug/pose', PoseStamped,
        #                                  queue_size=1)
        self.records = []
        self.sub_checkerboard_pose = rospy.Subscriber(
            '~input/checkerboard_pose', PoseStamped,
            self._cb_checkerboard_pose)
        self.timer_save = rospy.Timer(rospy.Duration(1), self._cb_save)

    def _cb_save(self, event):
        with open(self.out_file, 'w') as f:
            pandas.DataFrame(self.records).to_csv(f, index=False)
            rospy.loginfo('Saved to: %s' % self.out_file)

    def _cb_checkerboard_pose(self, msg):
        try:
            twist = self.tf_listener.lookupTwist(
                'base', msg.header.frame_id, msg.header.stamp,
                averaging_interval=rospy.Duration(1))
        except Exception as e:
            rospy.logerr(e)
            return
        velocity = np.linalg.norm(twist[0])
        if velocity > 0.1:
            rospy.logwarn('Arm is moving too fast: %f [m/s], so skipping.' %
                          velocity)
            return

        # base -> camera
        try:
            self.tf_listener.waitForTransform(
                'base', msg.header.frame_id, msg.header.stamp,
                timeout=rospy.Duration(0.01))
        except Exception as e:
            rospy.logerr(e)
            return
        translation, rotation = self.tf_listener.lookupTransform(
            'base', msg.header.frame_id, msg.header.stamp)
        translation = tf.transformations.translation_matrix(translation)
        rotation = tf.transformations.quaternion_matrix(rotation)
        matrix_base_to_camera = translation.dot(rotation)

        # camera -> checkerboard
        translation = msg.pose.position
        translation = translation.x, translation.y, translation.z
        translation = tf.transformations.translation_matrix(translation)
        rotation = msg.pose.orientation
        rotation = rotation.x, rotation.y, rotation.z, rotation.w
        rotation = tf.transformations.quaternion_matrix(rotation)
        matrix_camera_to_checkerboard = translation.dot(rotation)

        # base -> checkerboard
        matrix_base_to_checkerboard = \
            matrix_base_to_camera.dot(matrix_camera_to_checkerboard)
        translation = tf.transformations.translation_from_matrix(
            matrix_base_to_checkerboard)
        quaternion = tf.transformations.quaternion_from_matrix(
            matrix_base_to_checkerboard)
        euler = tf.transformations.euler_from_matrix(
            matrix_base_to_checkerboard)

        index = len(self.records)
        rospy.loginfo('[%d] translation: %s rotation: %s' %
                      (index, translation, rotation))
        self.records.append(dict(
            position_x=translation[0],
            position_y=translation[1],
            position_z=translation[2],
            quaternion_x=quaternion[0],
            quaternion_y=quaternion[1],
            quaternion_z=quaternion[2],
            quaternion_w=quaternion[3],
            euler_r=euler[0],
            euler_p=euler[1],
            euler_y=euler[2],
        ))

        # DEBUG
        # pose_msg = PoseStamped()
        # pose_msg.header.stamp = msg.header.stamp
        # pose_msg.header.frame_id = 'base'
        # pose_msg.pose.position.x = translation[0]
        # pose_msg.pose.position.y = translation[1]
        # pose_msg.pose.position.z = translation[2]
        # pose_msg.pose.orientation.x = rotation[0]
        # pose_msg.pose.orientation.y = rotation[1]
        # pose_msg.pose.orientation.z = rotation[2]
        # pose_msg.pose.orientation.w = rotation[3]
        # self.pub_debug.publish(pose_msg)


if __name__ == '__main__':
    rospy.init_node('test_hand_eye_coordination')
    app = TestHandEyeCoordination()
    rospy.spin()
