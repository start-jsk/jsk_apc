#!/usr/bin/env python
import numpy as np
import tf 
from geometry_msgs.msg import Transform, TransformStamped, Vector3, Quaternion
from sensor_msgs import point_cloud2
import matplotlib.pyplot as plt
import time


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


def visualize_cloud(cloud):
    gen = point_cloud2.read_points(cloud, skip_nans=False, field_names=("x", "y", "z"))
    points = [point for point in gen]

    arr = np.array(points).reshape((cloud.height, cloud.width, 3))
    plt.imsave('base_img.png', arr)


def timing(wrapped):
    def inner(*args, **kwargs):
        start = time.time()
        ret = wrapped(*args, **kwargs)
        end = time.time()
        print '{0} elapsed time {1}'.format(wrapped.__name__, end - start)
        return ret
    return inner

