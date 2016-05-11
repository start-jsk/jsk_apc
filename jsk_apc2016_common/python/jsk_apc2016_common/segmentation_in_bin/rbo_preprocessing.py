#!/usr/bin/env python

import jsk_apc2016_common.segmentation_in_bin.\
        segmentation_in_bin_helper as helper
import numpy as np
from sensor_msgs import point_cloud2
from matplotlib.path import Path
from tf2_geometry_msgs import do_transform_point


def get_spatial_img(bb_base2camera, cloud, target_bin):
    """
    :param bb_base2camera: transform from the boundingbox's frame
                            to camera's frame
    :type  bb_base2camera: transformStamped
    :param cloud:
    :type cloud: PointCloud2
    :param target_bin:
    :type target_bin: Bin
    :param dist_img      : distance of a point from shelf (mm)
    :type  dist_img      : np.array, dtype=uint8
    """
    bb_base2camera_mat = helper.tfmat_from_tf(bb_base2camera)
    bbox2bb_base_mat = helper.inv_tfmat(
        helper.tfmat_from_bbox(target_bin.bbox))
    bbox2camera_mat = np.dot(bbox2bb_base_mat, bb_base2camera_mat)
    bbox2camera = helper.tf_from_tfmat(bbox2camera_mat)

    dist_list, height_list = get_spatial(
            cloud, target_bin.bbox, bbox2camera,
            target_bin.camera_direction)

    cloud_shape = (cloud.height, cloud.width)
    dist_img = np.array(dist_list).reshape(cloud_shape)
    height_img = np.array(height_list).reshape(cloud_shape)
    # scale to mm from m
    dist_img = (dist_img * 1000).astype(np.uint8)
    height_img = (height_img * 2)  # adopting RBO's metric
    height_img[height_img == 0] = -1
    return dist_img, height_img


@helper.timing
def get_spatial(cloud, bbox, trans, direction):
    """
    :param trans: transformation from the cloud' parent frame
        to the bbox's center
    :param direction: on the axis of "direction", the distance from shelf is
        calculated only from a wall in positive coordinate of the axis
        Currently only x axis is supported)
    """
    assert direction == 'x'

    # represent a point in bounding box's frame
    # http://answers.ros.org/question/9103/how-to-transform-pointcloud2-with-tf/
    cloud_transformed = helper.do_transform_cloud(cloud, trans)
    points = point_cloud2.read_points(
            cloud_transformed,
            skip_nans=False,
            field_names=("x", "y", "z"))

    def get_spatial_feature(point, bbox):
        def d2wall(coord, width):
            if coord >= 0 and coord < width/2:
                return abs(width/2 - coord)
            elif coord < 0 and abs(coord) < width/2:
                return abs(coord + width/2)
            else:
                return 0

        def d2front(coord, width):
            if abs(coord) <= width/2:
                return width/2 - coord
            else:
                return 0

        d2wall_x_back = d2front(point[0], float(bbox.dimensions.x))
        d2wall_y = d2wall(point[1], float(bbox.dimensions.y))
        d2wall_z = d2wall(point[2], float(bbox.dimensions.z))
        d2wall_z_bottom = d2front(-point[2], float(bbox.dimensions.z))
        return (min(d2wall_x_back, d2wall_y, d2wall_z), d2wall_z_bottom)

    spatial_features = [get_spatial_feature(point, bbox) for point in points]
    distance_features, height_features = zip(*spatial_features)
    return distance_features, height_features


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
