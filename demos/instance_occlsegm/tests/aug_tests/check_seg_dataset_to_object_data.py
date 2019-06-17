import numpy as np

import instance_occlsegm_lib


def view_object_data(object_data):
    for objd in object_data:
        lbl = instance_occlsegm_lib.image.mask_to_lbl(
            objd['mask'], objd['label'])
        lbl_viz = instance_occlsegm_lib.datasets.visualize_label(
            lbl, objd['img'], seg_dataset.class_names)
        viz = np.hstack((objd['img'], lbl_viz))
        instance_occlsegm_lib.io.imshow(viz)
        if instance_occlsegm_lib.io.waitkey() == ord('q'):
            break


seg_dataset = instance_occlsegm_lib.datasets.apc.JskARC2017DatasetV3(
    split='valid')
random_state = np.random.RandomState(1)
object_data = instance_occlsegm_lib.aug.seg_dataset_to_object_data(
    seg_dataset, random_state, ignore_labels=[-1, 0, 41])
view_object_data(object_data)
