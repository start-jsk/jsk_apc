import bisect
import math
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.ndimage import center_of_mass


def _reliable_region_center(lbl, reliable_pts_ratio, prob_threshold):
    lbl_sorted = np.sort(lbl[lbl != 0])
    reliable_thre = lbl_sorted[int(len(lbl_sorted) *
                                   (1 - reliable_pts_ratio))]
    cy, cx = center_of_mass(lbl >= reliable_thre)
    if lbl[int(round(cy)), int(round(cx))] < prob_threshold:
        # In case centroid is outside instance/affordance
        cy, cx = center_of_mass(lbl == np.max(lbl))
    return np.array([cx, cy])


def _get_pinch_poses(
    prim_pinch_line,
    prim_pinch_pt,
    prim_pinch_yaw,
    labels,
    label_clustersss,
    prob_threshold,
    reliable_pts_ratio,
    instance_ids,
):
    poses = []
    # [[instance id, array([x, y]), yaw angle], ...]
    probs = []

    for i, inst_clusterss in enumerate(label_clustersss):
        deg_reso = 180 // len(inst_clusterss)
        for degree_id, clusters in enumerate(inst_clusterss):
            # Currently, following skipping is unnecessary
            # # Skip this deg channel in pinch label if it has no good pinch
            # if (inst_clusterss[degree_id] < 1).all():
            #     continue
            pinch_yaws = [degree_id * deg_reso, degree_id * deg_reso + 180]

            # For debug
            # import grasp_fusion_lib
            # grasp_fusion_lib.io.imshow(
            #     grasp_fusion_lib.image.label2rgb(clusters + 1)
            # )
            # grasp_fusion_lib.io.waitkey(0)

            # Check clusters of good pinch
            for s in np.unique(clusters):
                # Skip unclustered points
                if s < 0:
                    continue
                # Get centroid of reliable region of assigned cluster
                pinch_mask = (clusters == s)
                lbl_masked = np.copy(labels[i][degree_id])
                lbl_masked[~pinch_mask] = 0
                # array([x, y])
                pinch_pt = _reliable_region_center(
                    lbl_masked, reliable_pts_ratio, prob_threshold)

                for pinch_yaw in pinch_yaws:
                    # Get rotation angle of primitive
                    # to align prim_pinch_line with assigned pinch_yaw
                    prim_rot_yaw = pinch_yaw - prim_pinch_yaw
                    while int(prim_rot_yaw / 360) > 0:
                        prim_rot_yaw -= 360
                    while int(prim_rot_yaw / 360) < 0:
                        prim_rot_yaw += 360
                    # Get position of origin of placed primitive
                    rad = math.radians(prim_rot_yaw)
                    rot_tran = np.array([[np.cos(rad), np.sin(rad)],
                                         [-np.sin(rad), np.cos(rad)]])
                    # array([x, y])
                    prim_pos = pinch_pt - np.dot(prim_pinch_pt, rot_tran)
                    # Check if primitive origin is inside image
                    H, W = pinch_mask.shape
                    if 0 <= prim_pos[1] < H and 0 <= prim_pos[0] < W:
                        # Sort poses by mean prob
                        prob = np.mean(lbl_masked)
                        idx = bisect.bisect_left(probs, prob)
                        probs.insert(idx, prob)
                        poses.insert(
                            idx, [instance_ids[i], prim_pos, prim_rot_yaw]
                        )

    # bisect sorts in ascending order, but we need descending order
    poses.reverse()
    return poses


def _get_suction_poses(
    prim_suc_pt,
    labels,
    label_clustersss,
    prob_threshold,
    reliable_pts_ratio,
    instance_ids,
):
    poses = []
    # [[instance id, array([x, y]), yaw angle], ...]
    probs = []

    for i, inst_clusterss in enumerate(label_clustersss):
        # Check clusters of good suction
        for s in np.unique(inst_clusterss[0]):
            # Skip unclustered points
            if s < 0:
                continue
            # Get centroid of reliable region of assigned cluster
            suc_mask = (inst_clusterss[0] == s)
            lbl_masked = np.copy(labels[i][0])
            lbl_masked[~suc_mask] = 0
            # array([x, y])
            suc_pt = _reliable_region_center(
                lbl_masked, reliable_pts_ratio, prob_threshold)

            # Get position of origin of placed primitive
            # so that prim_suc_pt accords with suc_pt.
            # array([x, y])
            prim_pos = suc_pt - prim_suc_pt
            # Check if primitive origin is inside image
            H, W = suc_mask.shape
            if 0 <= prim_pos[1] < H and 0 <= prim_pos[0] < W:
                # Sort poses by mean prob
                prob = np.mean(lbl_masked)
                idx = bisect.bisect_left(probs, prob)
                probs.insert(idx, prob)
                poses.insert(idx, [instance_ids[i], prim_pos, 0])

    # bisect sorts in ascending order, but we need descending order
    poses.reverse()
    return poses


def _rotational_search(
    degree_id,
    deg_reso,
    clusters,
    lbl_center,
    lbl_rotated,
    prim_center_pt,
    prim_rotated_pt,
    prim_pinch_yaw,
    prob_threshold,
    reliable_pts_ratio,
):
    poses_no_inst = []
    # [[array([x, y]), yaw angle], ...]
    probs = []

    # Variables to check assigned yaw angle with range
    orig_pinch_yaw = degree_id * deg_reso
    yaw_range = deg_reso // 2
    step = yaw_range // 5
    add_yaw_list_cand = [sorted(range(-yaw_range, yaw_range + 1, step),
                                key=abs),
                         sorted(range(180 - yaw_range, 181 + yaw_range, step),
                                key=lambda x: abs(x - 180))]
    # [[0, -n, n, ..., -N, N], [180, 180-n, 180+n, ..., 180-N, 180+N]]

    # Check clusters of good pinch/suction
    for s in np.unique(clusters):
        # Skip unclustered points
        if s < 0:
            continue
        # Get centroid of reliable region of assigned cluster
        mask = (clusters == s)
        lbl_masked = np.copy(lbl_center)
        lbl_masked[~mask] = 0
        # array([x, y])
        center_pt = _reliable_region_center(
            lbl_masked, reliable_pts_ratio, prob_threshold)

        for add_yaw_list in add_yaw_list_cand:
            # Check until primitive is matched
            for add_yaw in add_yaw_list:
                pinch_yaw = orig_pinch_yaw + add_yaw
                # Get rotation angle of primitive
                # to align prim_pinch_line with assigned pinch_yaw
                prim_rot_yaw = pinch_yaw - prim_pinch_yaw
                while int(prim_rot_yaw / 360) > 0:
                    prim_rot_yaw -= 360
                while int(prim_rot_yaw / 360) < 0:
                    prim_rot_yaw += 360
                # Get position of rotated pt when primitive is placed
                # so that prim_center_pt accords with center_pt
                # and prim_pinch_line is aligned with pinch_yaw
                rad = math.radians(prim_rot_yaw)
                rot_tran = np.array([[np.cos(rad), np.sin(rad)],
                                     [-np.sin(rad), np.cos(rad)]])
                # array([x, y])
                rotated_pt = center_pt + \
                    np.dot((prim_rotated_pt - prim_center_pt), rot_tran)
                rotated_pt = rotated_pt.astype(int)
                # Check if assigned point is suctionable
                H, W = lbl_rotated.shape
                if not (0 <= rotated_pt[1] < H and 0 <= rotated_pt[0] < W):
                    continue
                rotated_pt_prob = lbl_rotated[rotated_pt[1]][rotated_pt[0]]
                if rotated_pt_prob >= prob_threshold:
                    # Get position of origin of placed primitive.
                    # array([x, y])
                    prim_pos = center_pt - np.dot(prim_center_pt, rot_tran)
                    if 0 <= prim_pos[1] < H and 0 <= prim_pos[0] < W:
                        # Sort poses by mean prob
                        prob = np.mean(lbl_masked) * rotated_pt_prob
                        probs.append(prob)
                        poses_no_inst.append([prim_pos, prim_rot_yaw])
                        break

    return poses_no_inst, probs


def _get_pinch_suc_poses(
    prim_pinch_line,
    prim_pinch_pt,
    prim_pinch_yaw,
    prim_suc_pt,
    labels_pinch,
    label_pinch_clustersss,
    labels_suction,
    prob_threshold,
    reliable_pts_ratio,
    instance_ids,
):
    poses = []
    # [[instance id, array([x, y]), yaw angle], ...]
    probs = []

    for i, inst_clusterss in enumerate(label_pinch_clustersss):
        deg_reso = 180 // len(inst_clusterss)
        for degree_id, clusters in enumerate(inst_clusterss):
            # Skip this deg channel in pinch label if it has no good pinch
            if (clusters < 0).all():
                continue

            new_poses, new_probs = _rotational_search(
                degree_id,
                deg_reso,
                clusters,
                labels_pinch[i][degree_id],
                labels_suction[i][0],
                prim_pinch_pt,
                prim_suc_pt,
                prim_pinch_yaw,
                prob_threshold,
                reliable_pts_ratio,
            )
            # Sort poses by mean prob
            for j, prob in enumerate(new_probs):
                idx = bisect.bisect_left(probs, prob)
                probs.insert(idx, prob)
                new_poses[j].insert(0, instance_ids[i])
                poses.insert(idx, new_poses[j])

    # bisect sorts in ascending order, but we need descending order
    poses.reverse()
    return poses


def _get_suc_then_pinch_poses(
    prim_pinch_line,
    prim_pinch_pt,
    prim_pinch_yaw,
    prim_suc_pt,
    labels_pinch,
    labels_suction,
    label_suc_clustersss,
    prob_threshold,
    reliable_pts_ratio,
    instance_ids,
):
    poses = []
    # [[instance id, array([x, y]), yaw angle], ...]
    probs = []

    for i, inst_clusterss in enumerate(label_suc_clustersss):
        deg_reso = 180 // len(labels_pinch[i])
        for degree_id, label_pinch in enumerate(labels_pinch[i]):
            # Skip this deg channel in pinch label if it has no good pinch
            if (label_pinch < prob_threshold).all():
                continue

            new_poses, new_probs = _rotational_search(
                degree_id,
                deg_reso,
                inst_clusterss[0],
                labels_suction[i][0],
                labels_pinch[i][degree_id],
                prim_suc_pt,
                prim_pinch_pt,
                prim_pinch_yaw,
                prob_threshold,
                reliable_pts_ratio,
            )
            # Sort poses by mean prob
            for j, prob in enumerate(new_probs):
                idx = bisect.bisect_left(probs, prob)
                probs.insert(idx, prob)
                new_poses[j].insert(0, instance_ids[i])
                poses.insert(idx, new_poses[j])

    # bisect sorts in ascending order, but we need descending order
    poses.reverse()
    return poses


def get_primitives_poses(
    primitives,
    heightmap,
    labels,
    affordances,
    cluster_tolerance=0.02,  # Max distance[m] in same cluster
    cluster_max_size=None,  # Max size[m^2] of cluster
    cluster_min_size=None,  # Min size[m^2] of cluster
    voxel_size=0.002,  # Size[m] of each pixel
    instance_label=None,  # Instance label image
    instance_bg_label=-1,  # Instance label of background
    prob_threshold=0.5,
    reliable_pts_ratio=0.25,
    print_time=False,
):
    assert len(labels) == len(affordances)
    if instance_label is None:
        # Set label image for test assuming whole image is one instance
        instance_label = np.full_like(labels[affordances.index('suction')],
                                      instance_bg_label + 1)

    ret_posess = []
    # [[[instance id, array([x, y]), yaw angle], ...] for first primitive, ...]

    # For debug
    if print_time:
        import time
        start = time.time()

    # Make label image for each instance &
    # euclidean clustering to reduce candidates
    inst_ids = sorted(np.unique(instance_label),
                      key=lambda x: np.sum((instance_label == x) *
                                           (heightmap > 0)),
                      reverse=True)
    # labels_sep_with_inst[affordance id][instance id][degree id]
    #     = mask of affordance in instance
    labels_sep_with_inst = []
    # whole_clusterssss[affordance id][instance id][degree id]
    #     = clusters of affordance in instance
    whole_clusterssss = []
    c_thre = cluster_tolerance / voxel_size  # Max distance in pixel
    for i, lbl in enumerate(labels):
        label_sep_with_inst = []
        lbl_clustersss = []
        for inst in inst_ids:
            if len(lbl.shape) == 2:
                lbl = lbl.reshape(lbl.shape[0], lbl.shape[1], 1)
            if len(lbl.shape) != 3:
                print("Invalid label")
                return
            label_each_inst = []
            clusterss = []
            for channel in range(lbl.shape[2]):
                if inst > instance_bg_label:
                    lbl_inst = np.copy(lbl[:, :, channel])
                    lbl_inst[~((instance_label == inst) *
                               (heightmap > 0))] = 0
                else:
                    # Background
                    lbl_inst = np.zeros_like(lbl[:, :, channel])
                # array([[y, x], [y, x], ...])
                good_pts = np.array(np.where(lbl_inst >= prob_threshold)).T
                clusters = np.full_like(lbl[:, :, channel], -1)
                if len(good_pts) > 1:
                    link_mat = linkage(good_pts,
                                       metric='euclidean', method='single')
                    if link_mat.shape[0] != 0:
                        # array([id of point1, id of point2, ...])
                        good_pts_lbl = fcluster(link_mat, c_thre,
                                                criterion='distance')
                        for i, pt in enumerate(good_pts):
                            clusters[pt[0], pt[1]] = good_pts_lbl[i]
                label_each_inst.append(lbl_inst)
                clusterss.append(clusters)
            label_sep_with_inst.append(label_each_inst)
            lbl_clustersss.append(clusterss)
        labels_sep_with_inst.append(label_sep_with_inst)
        whole_clusterssss.append(lbl_clustersss)

    # Limit cluster size
    if (cluster_max_size is not None) or (cluster_min_size is not None):
        if cluster_max_size is not None:
            c_max_pix = cluster_max_size / (voxel_size ** 2)
            # Max size in pixel
        if cluster_min_size is not None:
            c_min_pix = cluster_min_size / (voxel_size ** 2)
            # Min size in pixel
        for i, clustersss in enumerate(whole_clusterssss):
            for j, clusterss in enumerate(clustersss):
                for k, clusters in enumerate(clusterss):
                    for s in np.unique(clusters):
                        # Skip unclustered points
                        if s < 0:
                            continue
                        # Get area of assigned cluster
                        mask = (clusters == s)
                        area = np.sum(mask)
                        if cluster_max_size is not None and area > c_max_pix:
                            clusters[mask] = -1
                        if cluster_min_size is not None and area < c_min_pix:
                            clusters[mask] = -1
                    whole_clusterssss[i][j][k] = clusters

    if print_time:
        elapsed_time = time.time() - start
        print("Time for clustering: {0}".format(elapsed_time) + "[sec]")

    lbl_pinch_idx = None
    lbl_pinch_sole_idx = None
    lbl_suc_idx = None
    for primitive in primitives:
        # Decord primitive
        try:
            prim_pinch_idx = next(i for i, v in enumerate(primitive['shapes'])
                                  if v['affordance'] == 'pinch')
            prim_pinch_order = primitive['shapes'][prim_pinch_idx]['order']
        except StopIteration:
            prim_pinch_idx = None
        try:
            prim_suc_idx = next(i for i, v in enumerate(primitive['shapes'])
                                if v['affordance'] == 'suction')
            prim_suc_order = primitive['shapes'][prim_suc_idx]['order']
        except StopIteration:
            prim_suc_idx = None
        if prim_pinch_idx is not None:
            # array([[x0, y0], [x1, y1]])
            prim_pinch_line = np.array(primitive['shapes'][prim_pinch_idx]
                                                ['points'])
            prim_pinch_line /= (voxel_size * 1000)
            prim_pinch_pt = np.sum(prim_pinch_line, axis=0) / 2.0
            vec = prim_pinch_line[1] - prim_pinch_line[0]
            prim_pinch_yaw = np.degrees(np.arctan2(vec[1], vec[0]))
            # 0 deg z-axis rotation in pinch label means horizontal pinch.
        if prim_suc_idx is not None:
            # array([x, y])
            prim_suc_pt = np.array(primitive['shapes'][prim_suc_idx]
                                            ['points'][0])
            prim_suc_pt /= (voxel_size * 1000)

        # Run appropriate matching function
        if (prim_pinch_idx is not None) and (prim_suc_idx is not None) and \
           (prim_pinch_order == prim_suc_order):
            if lbl_pinch_idx is None:
                lbl_pinch_idx = affordances.index('pinch')
            if lbl_suc_idx is None:
                lbl_suc_idx = affordances.index('suction')
            poses = _get_pinch_suc_poses(
                prim_pinch_line,
                prim_pinch_pt,
                prim_pinch_yaw,
                prim_suc_pt,
                labels_sep_with_inst[lbl_pinch_idx],
                whole_clusterssss[lbl_pinch_idx],
                labels_sep_with_inst[lbl_suc_idx],
                prob_threshold,
                reliable_pts_ratio,
                inst_ids,
            )
        elif (prim_pinch_idx is not None) and (prim_suc_idx is not None) and \
             (prim_pinch_order > prim_suc_order):
            if lbl_pinch_sole_idx is None:
                lbl_pinch_sole_idx = affordances.index('pinch_sole')
            if lbl_suc_idx is None:
                lbl_suc_idx = affordances.index('suction')
            poses = _get_suc_then_pinch_poses(
                prim_pinch_line,
                prim_pinch_pt,
                prim_pinch_yaw,
                prim_suc_pt,
                labels_sep_with_inst[lbl_pinch_sole_idx],
                labels_sep_with_inst[lbl_suc_idx],
                whole_clusterssss[lbl_suc_idx],
                prob_threshold,
                reliable_pts_ratio,
                inst_ids,
            )
        elif prim_pinch_idx is not None:
            if lbl_pinch_idx is None:
                lbl_pinch_idx = affordances.index('pinch')
            poses = _get_pinch_poses(
                prim_pinch_line,
                prim_pinch_pt,
                prim_pinch_yaw,
                labels_sep_with_inst[lbl_pinch_idx],
                whole_clusterssss[lbl_pinch_idx],
                prob_threshold,
                reliable_pts_ratio,
                inst_ids
            )
        elif prim_suc_idx is not None:
            if lbl_suc_idx is None:
                lbl_suc_idx = affordances.index('suction')
            poses = _get_suction_poses(
                prim_suc_pt,
                labels_sep_with_inst[lbl_suc_idx],
                whole_clusterssss[lbl_suc_idx],
                prob_threshold,
                reliable_pts_ratio,
                inst_ids
            )
        else:
            print("Invalid primitive")
        ret_posess.append(poses)

    # For debug
    if print_time:
        elapsed_time = time.time() - start
        print("Whole execution: {0}".format(elapsed_time) + "[sec]")

    return ret_posess
