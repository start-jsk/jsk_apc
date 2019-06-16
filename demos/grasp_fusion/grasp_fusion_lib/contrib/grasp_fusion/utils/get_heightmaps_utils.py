from __future__ import division

import numpy as np
import skimage.morphology


def get_heightmap(
    color_img,
    depth_img,
    bg_color_img,
    bg_depth_img,
    cam_intrinsics,
    cam_pose,
    grid_origin,
    grid_rot,
    suction_img=None,
    voxel_size=0.002,  # Size[m] of each height map pixel
    suction_cval=0,
):

    # Do background subtraction
    fg_mask_color = ~(
        np.sum(np.abs(color_img - bg_color_img) < 0.3, 2) == 3)
    fg_mask_depth = (bg_depth_img != 0) & (
        np.abs(depth_img - bg_depth_img) > 0.02)
    fg_mask = fg_mask_color | fg_mask_depth

    # Project depth into camera space
    pix_x, pix_y = np.meshgrid(
        np.arange(depth_img.shape[1]),
        np.arange(depth_img.shape[0]),
    )
    cam_x = (pix_x - cam_intrinsics[0, 2]) * \
        depth_img / cam_intrinsics[0, 0]
    cam_y = (pix_y - cam_intrinsics[1, 2]) * \
        depth_img / cam_intrinsics[1, 1]
    cam_z = depth_img
    cam_pts = np.array([
        cam_x.flatten(),
        cam_y.flatten(),
        cam_z.flatten(),
    ])  # shape: (3, 307200)

    # Transform points to world coordinates
    world_pts = (
        np.dot(cam_pose[:3, :3], cam_pts) +
        np.tile(cam_pose[:3, 3][:, None], (1, cam_pts.shape[1]))
    ).T  # shape: (307200, 3)

    # Transform points to grid coordinates
    grid_pts = np.dot((world_pts - grid_origin), grid_rot)

    # Get height map
    heightmap = np.zeros((200, 300))
    grid_mapping = np.array([
        np.round(grid_pts[:, 0] / voxel_size),
        np.round(grid_pts[:, 1] / voxel_size),
        grid_pts[:, 2],
    ]).T  # shape: (307200, 3) # NOQA

    # Compute height map color
    valid_pix = (
        # grid_mapping[:, 0]: grid_voxel_x
        # grid_mapping[:, 1]: grid_voxel_y
        (grid_mapping[:, 0] >= 0) & (grid_mapping[:, 0] < 300) &
        (grid_mapping[:, 1] >= 0) & (grid_mapping[:, 1] < 200)
    )
    # ...  and grid_mapping[:, 2] > 0  # shape: (307200,)
    color_pts = np.array([
        color_img[:, :, 0].flatten(),
        color_img[:, :, 1].flatten(),
        color_img[:, :, 2].flatten(),
    ]).T  # shape: (307200, 3) # NOQA
    heightmap_color = np.zeros((200 * 300, 3))
    heightmap_color[
        grid_mapping[valid_pix, 1].astype(int) * heightmap.shape[1] +
        grid_mapping[valid_pix, 0].astype(int),
    ] = color_pts[valid_pix]

    heightmap_suction = None
    if suction_img is not None:
        suction_pts = suction_img.flatten()
        heightmap_suction = np.full(
            (200 * 300), suction_cval, dtype=suction_pts.dtype
        )
        heightmap_suction[
            grid_mapping[valid_pix, 1].astype(int) * heightmap.shape[1] +
            grid_mapping[valid_pix, 0].astype(int),
        ] = suction_pts[valid_pix]

    # Compute real height map with background subtraction
    valid_pix = (
        (grid_mapping[:, 0] >= 0) & (grid_mapping[:, 0] < 300) &
        (grid_mapping[:, 1] >= 0) & (grid_mapping[:, 1] < 200) &
        (grid_mapping[:, 2] >= 0)
    )  # shape: (307200,)
    valid_depth = fg_mask & (cam_z != 0)  # shape: (480, 640) # NOQA
    grid_mapping = grid_mapping[valid_pix & valid_depth.reshape(-1)]
    heightmap[
        grid_mapping[:, 1].astype(np.int),
        grid_mapping[:, 0].astype(np.int),
    ] = grid_mapping[:, 2]

    # Find missing depth and project background depth into camera space
    missing_depth = np.logical_and(depth_img == 0, bg_depth_img > 0)
    pix_x, pix_y = np.meshgrid(np.arange(640), np.arange(480))
    cam_x = (pix_x - cam_intrinsics[0, 2]) * \
        bg_depth_img / cam_intrinsics[0, 0]
    cam_y = (pix_y - cam_intrinsics[1, 2]) * \
        bg_depth_img / cam_intrinsics[1, 1]
    cam_z = bg_depth_img
    missing_cam_pts = np.array([
        cam_x[missing_depth],
        cam_y[missing_depth],
        cam_z[missing_depth],
    ])
    missing_world_pts = (
        cam_pose[:3, :3].dot(missing_cam_pts) +
        np.tile(cam_pose[:3, 3][:, None],
                (1, missing_cam_pts.shape[1]))
    ).T
    missing_grid_pts = np.dot((missing_world_pts - grid_origin), grid_rot)

    # Get missing depth height map
    missing_heightmap = np.zeros(heightmap.shape[:2], dtype=np.bool)
    grid_mapping = np.array([
        np.round(missing_grid_pts[:, 0] / voxel_size),
        np.round(missing_grid_pts[:, 1] / voxel_size),
        missing_grid_pts[:, 2],
    ]).T
    valid_pix = (
        (grid_mapping[:, 0] >= 0) & (grid_mapping[:, 0] < 300) &
        (grid_mapping[:, 1] >= 0) & (grid_mapping[:, 1] < 200)
    )
    grid_mapping = grid_mapping[valid_pix]
    missing_heightmap[
        grid_mapping[:, 1].astype(np.int),
        grid_mapping[:, 0].astype(np.int),
    ] = 1

    noise_pix = ~skimage.morphology.remove_small_objects(
        missing_heightmap > 0, min_size=50, connectivity=1)
    missing_heightmap[noise_pix] = 0

    # Denoise height map
    noise_pix = ~skimage.morphology.remove_small_objects(
        heightmap > 0, min_size=50, connectivity=1)
    heightmap[noise_pix] = 0

    return heightmap_color, heightmap, missing_heightmap, heightmap_suction


def heightmap_postprocess(heightmap_color, heightmap, missing_heightmap):
    heightmap_color = heightmap_color.reshape(
        heightmap.shape[0], heightmap.shape[1], 3
    )

    # Height cannot exceed 30cm above bottom of tote
    heightmap = np.minimum(heightmap, np.ones(heightmap.shape) * 0.3)

    # Fill in missing depth holes (assume height of 3cm)
    heightmap[(heightmap == 0) & missing_heightmap] = 0.03

    # Add extra padding to height map and reprojected color images
    # Padding: +12px both side y-axis, +10px both side x-axis
    # CAUTION: This padding influences 3D pose calculation
    # in primitive_matching
    color_data = np.zeros((224, 320, 3), dtype=np.uint8)
    depth_data = np.zeros((224, 320), dtype=np.uint16)
    color_data[12:212, 10:310, :] = heightmap_color * 255
    depth_data[12:212, 10:310] = heightmap * 10000

    return color_data, depth_data
