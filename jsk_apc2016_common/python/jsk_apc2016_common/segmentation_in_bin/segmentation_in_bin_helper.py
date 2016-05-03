#!/usr/bin/env python

import numpy as np
import tf
import rospy
from geometry_msgs.msg import (
        Transform, TransformStamped,
        Quaternion, Vector3,
        Point)
import time
from sensor_msgs.point_cloud2 import read_points, create_cloud
import PyKDL


def quaternion(rot):
    return Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])


def vector3(vec):
    return Vector3(x=vec[0], y=vec[1], z=vec[2])


def point(vec):
    return Point(x=vec[0], y=vec[1], z=vec[2])


def list_from_point(point):
    return [point.x, point.y, point.z]


def mat_from_pos(vec):
    vec_arr = np.array((vec.x, vec.y, vec.z))
    return tf.transformations.translation_matrix(vec_arr)


def mat_from_orient(orient):
    orient_arr = np.array((orient.x, orient.y, orient.z, orient.w))
    return tf.transformations.quaternion_matrix(orient_arr)


def tfmat_from_bbox(bbox):
    trans = mat_from_pos(bbox.pose.position)
    rot = mat_from_orient(bbox.pose.orientation)
    return np.dot(trans, rot)


def tfmat_from_tf(tf):
    trans = mat_from_pos(tf.transform.translation)
    rot = mat_from_orient(tf.transform.rotation)
    return np.dot(trans, rot)


def tf_from_tfmat(mat):
    tf_stamped = TransformStamped()
    _tf = Transform()

    t_vec = tf.transformations.translation_from_matrix(mat)
    _tf.translation.x = t_vec[0]
    _tf.translation.y = t_vec[1]
    _tf.translation.z = t_vec[2]

    q_vec = tf.transformations.quaternion_from_matrix(mat)
    _tf.rotation.x = q_vec[0]
    _tf.rotation.y = q_vec[1]
    _tf.rotation.z = q_vec[2]
    _tf.rotation.w = q_vec[3]
    tf_stamped.transform = _tf
    # header and child_frame_id needs to be defined
    return tf_stamped


def inv_tfmat(mat):
    assert mat.shape == (4, 4)

    rot = mat[:3, :3]
    trans = mat[:3, 3]
    ret = np.zeros((4, 4))
    ret[:3, :3] = np.transpose(rot)
    ret[:3, 3] = - np.dot(np.transpose(rot), trans)
    ret[3, 3] = 1
    return ret


def timing(wrapped):
    def inner(*args, **kwargs):
        start = time.time()
        ret = wrapped(*args, **kwargs)
        end = time.time()
        rospy.loginfo('{0} elapsed time {1}'.format(wrapped.__name__, end - start))
        return ret
    return inner


# patched version of tf2_sensor_msgs.py
def transform_to_kdl(t):
    return PyKDL.Frame(
            PyKDL.Rotation.Quaternion(
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w),
            PyKDL.Vector(
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z))


def do_transform_cloud(cloud, transform):
    t_kdl = transform_to_kdl(transform)
    points_out = []
    for p_in in read_points(cloud):
        coord_out = t_kdl * PyKDL.Vector(p_in[0], p_in[1], p_in[2])
        p_out = (coord_out.x(), coord_out.y(), coord_out.z()) + p_in[3:]
        points_out.append(p_out)
    res = create_cloud(transform.header, cloud.fields, points_out)
    return res
