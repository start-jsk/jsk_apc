#!/usr/bin/env python


def view_labels_with_primitive(label_pinch, label_suction,
                               primitive, prim_poses, background):
    import grasp_fusion_lib
    import numpy as np
    import skimage.io

    # # Consts
    # MM_TO_PIXEL = 0.5  # 1 mm is that many pixel

    # # Read and convert primitive
    # suc_idx = next(i for i, v in enumerate(primitive['shapes'])
    #                if v['affordance'] == 'suction')
    # # array([x, y])
    # prim_suc_pt = np.array(primitive['shapes'][suc_idx]['points'][0])
    # prim_suc_pt *= MM_TO_PIXEL

    for pose in prim_poses:
        # # Generate suction point on label img from matched primitive
        # rad = math.radians(pose[2])
        # rot_tran = np.array([[np.cos(rad), np.sin(rad)],
        #                      [-np.sin(rad), np.cos(rad)]])
        # # array([x, y])
        # suc_pt = pose[1] + np.dot(prim_suc_pt, rot_tran)

        # Create label img
        # label 1: good suction, label 2: good pinch in all deg channels,
        # label 3: suction point of matched primitive
        label_pinch_binary = np.where(label_pinch < 0.5, 0, 1)
        label_pinch_sum = np.sum(label_pinch_binary, axis=2)
        label_pinch_sum_binary = np.where(label_pinch_sum < 0.5, 0, 2)
        label_suction_binary = np.where(label_suction < 0.5, 0, 1)
        label_pinch_and_suc = label_pinch_sum_binary + label_suction_binary
        label_pinch_and_suc = np.where(label_pinch_and_suc > 1.5, 2,
                                       label_pinch_and_suc)
        label_with_point = np.copy(label_pinch_and_suc)
        # rr, cc = skimage.draw.circle(suc_pt[1], suc_pt[0], 5)
        rr, cc = skimage.draw.circle(pose[1][1], pose[1][0], 5)
        label_with_point[rr, cc] = 3
        viz = grasp_fusion_lib.image.label2rgb(
            label_with_point,
            background,
            label_names=[None, None, None, None],
        )
        grasp_fusion_lib.io.imshow(viz)
        grasp_fusion_lib.io.waitkey()


def main():
    from grasp_fusion_lib.contrib import grasp_fusion
    from grasp_fusion_lib.contrib.grasp_fusion.utils import \
        get_primitives_poses
    import os
    import os.path as osp
    import yaml

    dataset_pinch = grasp_fusion.datasets.PinchDataset('train')
    dataset_suction = grasp_fusion.datasets.SuctionDataset('train')
    color_pinch, depth_pinch, label_pinch = dataset_pinch[0]
    color_suction, depth_suction, label_suction = dataset_suction[0]

    primitives = []
    # Pinch only
    with open(osp.join(os.getcwd(),
                       'primitive_samples/pinch30mm.yaml')) as f:
        primitives.append(yaml.load(f))
    # Suction only
    with open(osp.join(os.getcwd(),
                       'primitive_samples/suction.yaml')) as f:
        primitives.append(yaml.load(f))
    # Pinch & Suction
    with open(osp.join(os.getcwd(),
                       'primitive_samples/pinch30mm_suction.yaml')) as f:
        primitives.append(yaml.load(f))
    # Suction then pinch
    with open(osp.join(os.getcwd(),
                       'primitive_samples/suction_then_pinch30mm.yaml')) as f:
        primitives.append(yaml.load(f))

    prim_posess = get_primitives_poses(
        primitives,
        depth_pinch,
        [label_pinch, label_pinch, label_suction],
        ['pinch', 'pinch_sole', 'suction'],
        print_time=True,
    )
    for i, poses in enumerate(prim_posess):
        print("primitive label: {0}".format(primitives[i]['label']))
        print("primitive poses: {0}".format(poses))
        view_labels_with_primitive(label_pinch, label_suction,
                                   primitives[i], poses, color_pinch)


if __name__ == '__main__':
    main()
