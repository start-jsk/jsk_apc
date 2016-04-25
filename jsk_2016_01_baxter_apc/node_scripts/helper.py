#!/usr/bin/env python
import numpy as np
import tf 
from geometry_msgs.msg import Transform, TransformStamped, Quaternion, Vector3, Point, PointStamped
import geometry_msgs.msg
from sensor_msgs import point_cloud2
import matplotlib.pyplot as plt
import time

def quaternion(rot):
    return Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])

def vector3(vec):
    return Vector3(x=vec[0], y=vec[1], z=vec[2])

def point(vec):
    return Point(x=vec[0], y=vec[1], z=vec[2])

def list_from_point(point):
    return [point.x, point.y, point.z]

def corner_point(initial_pos, dimensions, depth, header, signs):
    return PointStamped(
            header=header,
            point=Point(
                    x=initial_pos[0]+signs[0]*depth*dimensions[0]/2,
                    y=initial_pos[1]+signs[1]*dimensions[1]/2,
                    z=initial_pos[2]+signs[2]*dimensions[2]/2))

def mat_from_pos(vec):
    vec_arr = np.array((vec.x, vec.y, vec.z))
    return tf.transformations.translation_matrix(vec_arr)

def mat_from_orient(orient):
    orient_arr =  np.array((orient.x, orient.y, orient.z, orient.w))
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

    t_vec= tf.transformations.translation_from_matrix(mat)
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

    rot = mat[:3,:3] 
    trans = mat[:3, 3]
    ret = np.zeros((4,4))
    ret[:3,:3] = np.transpose(rot)
    ret[:3,3] = - np.dot(np.transpose(rot), trans)
    ret[3,3] = 1
    return ret

def timing(wrapped):
    def inner(*args, **kwargs):
        start = time.time()
        ret = wrapped(*args, **kwargs)
        end = time.time()
        print '{0} elapsed time {1}'.format(wrapped.__name__, end - start)
        return ret
    return inner
