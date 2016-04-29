#!/usr/bin/env python

import numpy as np
from matplotlib.path import Path
import jsk_apc2016_common.segmentation_helper as helper
from tf2_geometry_msgs import do_transform_point


def get_mask_img(transform, target_bin, camera_model):
    """
    :param point: point that is going to be transformed
    :type point: PointStamped
    :param transform: camera_frame -> bbox_frame
    :type transform: Transform
    """
    # check frame_id of a point and transform just in case
    assert camera_model.tf_frame == transform.header.frame_id
    assert target_bin.bbox.header.frame_id == transform.child_frame_id

    transformed_list = [
            do_transform_point(corner, transform)
            for corner in target_bin.corners]
    projected_points = project_points(transformed_list, camera_model)

    # generate an polygon that covers the region
    path = Path(projected_points)
    x, y = np.meshgrid(
            np.arange(camera_model.width),
            np.arange(camera_model.height))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    mask_img = path.contains_points(
            points).reshape(
                    camera_model.height, camera_model.width
                ).astype('bool')
    return mask_img


def project_points(points, camera_model):
    """
    :param points: list of geometry_msgs.msg.PointStamped
    :type list of stamped points :
    :param projected_points: list of camera_coordinates
    :type  projected_points: (u, v)

    The frames of the points and the camera_model are same.
    """
    # generate mask iamge
    for point in points:
        if point.header.frame_id != camera_model.tf_frame:
            raise ValueError('undefined')
    if len(points) != 4:
        raise ValueError('undefined')

    projected_points = []
    for point in points:
        projected_points.append(
                camera_model.project3dToPixel(
                        helper.list_from_point(point.point)
                    )
            )
    return projected_points
