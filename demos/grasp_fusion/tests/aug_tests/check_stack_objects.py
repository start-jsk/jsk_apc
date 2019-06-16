import numpy as np
import six

import grasp_fusion_lib


seg_dataset = grasp_fusion_lib.datasets.apc.JskARC2017DatasetV3(split='valid')
random_state = np.random.RandomState(1)
object_data = grasp_fusion_lib.aug.seg_dataset_to_object_data(
    seg_dataset, random_state, ignore_labels=[-1, 0, 41])

for i in six.moves.range(len(seg_dataset)):
    img, lbl = seg_dataset.get_example(i)
    lbl_suc = np.empty(img.shape[:2], dtype=np.int32)
    lbl_suc.fill(-1)

    lbl[lbl == 0] = -1  # ignore unknown objects
    stacked = grasp_fusion_lib.aug.stack_objects(
        img, lbl, object_data, region_label=41, random_state=random_state)

    lbl = grasp_fusion_lib.datasets.visualize_label(
        lbl, img, seg_dataset.class_names)
    class_names_suc = ['no_suction', 'suction']
    lbl_suc = grasp_fusion_lib.datasets.visualize_label(
        lbl_suc, img, class_names_suc)
    lbl_aug = grasp_fusion_lib.datasets.visualize_label(
        stacked['lbl'], stacked['img'], seg_dataset.class_names)
    viz = np.vstack((np.hstack((img, lbl)),
                     np.hstack((stacked['img'], lbl_aug))))
    viz = grasp_fusion_lib.image.resize(viz, height=500)
    grasp_fusion_lib.io.imshow(viz)
    if grasp_fusion_lib.io.waitkey() == ord('q'):
        break
