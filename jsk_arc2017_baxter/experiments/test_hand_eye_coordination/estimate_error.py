#!/usr/bin/env python

import math
import os.path as osp

import tf.transformations as tff

import pandas
import numpy as np


here = osp.dirname(osp.abspath(__file__))


def stat_poses(csv_file):
    print('%s %s %s' % ('>' * 20, csv_file, '<' * 20))
    df = pandas.DataFrame.from_csv(csv_file, index_col=None)

    # Compute average of transformation matrix
    # ========================================
    n_fused = 0
    matrix_fused = None  # average matrix from base -> checkerboard pose
    for index, row in df.iterrows():
        # base -> pointIn (sensed calibboard pose)
        pos = row['position_x'], row['position_y'], row['position_z']
        matrix_pos = tff.translation_matrix(pos)
        rot = row['quaternion_x'], row['quaternion_y'], row['quaternion_z'], row['quaternion_w']
        matrix_rot = tff.quaternion_matrix(rot)
        matrix_pose = matrix_pos.dot(matrix_rot)

        if n_fused == 0:
            matrix_fused = matrix_pose
            n_fused += 1
            continue

        # Fuse transformation matrix
        # --------------------------
        # base -> pointFused (fused calibboard pose)
        # matrix_fused

        # pointFused -> pointIn = pointFused -> base -> pointIn
        matrix_rel = matrix_pose.dot(tff.inverse_matrix(matrix_fused))

        # weight of the pointIn is 1 / (n_fused + 1)
        weight_of_new_point = 1. / (n_fused + 1)

        # midpoint of rotation
        angle, direction, point = tff.rotation_from_matrix(matrix_rel)
        matrix_rel_rot_mid = tff.rotation_matrix(
            angle * weight_of_new_point, direction, point)
        # midpoint of translation
        trans_rel = tff.translation_from_matrix(matrix_rel)
        matrix_rel_pos_mid = tff.translation_matrix(
            [x * weight_of_new_point for x in trans_rel])
        # relative transformation for midpoint from pointFused
        matrix_rel_mid = matrix_rel_pos_mid.dot(matrix_rel_rot_mid)

        # base -> pointFused_new
        matrix_fused = matrix_rel_mid.dot(matrix_fused)
        n_fused += 1
        # -------------------------------------------------------------------------
    assert n_fused == len(df)

    print('%s Average %s' % ('-' * 30, '-' * 30))
    print('N fused: %d' % n_fused)
    print('Matrix: \n%s' % matrix_fused)
    trans = tff.translation_from_matrix(matrix_fused)
    print('Position: {}'.format(trans))
    pos_keys = ['position_%s' % x for x in 'xyz']
    print('Position (simple): {}'.format(df.mean()[pos_keys].values))
    euler = tff.euler_from_matrix(matrix_fused)
    print('Orientation: {}'.format(euler))

    # Compute variance of transformation matrix
    # =========================================

    N = len(df)
    keys = ['x', 'y', 'z', 'angle']
    variance = {k: 0 for k in keys}
    for index, row in df.iterrows():
        # base -> pointIn (sensed calibboard pose)
        pos = row['position_x'], row['position_y'], row['position_z']
        matrix_pos = tff.translation_matrix(pos)
        rot = row['quaternion_x'], row['quaternion_y'], row['quaternion_z'], row['quaternion_w']
        matrix_rot = tff.quaternion_matrix(rot)
        matrix_pose = matrix_pos.dot(matrix_rot)

        # pointFused -> pointIn = pointFused -> base -> pointIn
        matrix_rel = matrix_pose.dot(tff.inverse_matrix(matrix_fused))
        # compute distance for translation/rotation
        delta_x, delta_y, delta_z = tff.translation_from_matrix(matrix_rel)
        delta_angle, _, _ = tff.rotation_from_matrix(matrix_rel)

        variance['x'] += delta_x ** 2
        variance['y'] += delta_y ** 2
        variance['z'] += delta_z ** 2
        variance['angle'] += delta_angle ** 2
    for k in keys:
        variance[k] /= (N - 1)

    print('%s Std Deviation (Variance) %s' % ('-' * 22, '-' * 22))
    for k in keys:
        print('%s: %f (%f)' % (k, variance[k], math.sqrt(variance[k])))
        if k != 'angle':
            print('{} (simple): {}'.format(k, df.std()['position_%s' % k]))


if __name__ == '__main__':
    for side in ['left', 'right']:
        csv_file = osp.join(here, '%s.csv' % side)
        stat_poses(csv_file)
