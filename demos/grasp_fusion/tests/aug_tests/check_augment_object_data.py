import numpy as np
import six

import grasp_fusion_lib


def view_object_data(object_data, object_data_aug, class_names):
    for objd, objd_aug in six.moves.zip(object_data, object_data_aug):
        lbl = grasp_fusion_lib.image.mask_to_lbl(objd['mask'], objd['label'])
        lbl_aug = grasp_fusion_lib.image.mask_to_lbl(
            objd_aug['mask'], objd_aug['label'])

        objd['img'] = grasp_fusion_lib.image.resize(
            objd['img'], size=200 * 200)
        lbl = grasp_fusion_lib.image.resize(
            lbl, size=200 * 200, interpolation=0)
        objd_aug['img'] = grasp_fusion_lib.image.resize(
            objd_aug['img'], size=200 * 200)
        lbl_aug = grasp_fusion_lib.image.resize(
            lbl_aug, size=200 * 200, interpolation=0)

        lbl = grasp_fusion_lib.datasets.visualize_label(
            lbl, objd['img'], class_names)
        lbl_aug = grasp_fusion_lib.datasets.visualize_label(
            lbl_aug, objd_aug['img'], class_names)
        viz = grasp_fusion_lib.image.tile([
            np.hstack((objd['img'], lbl)),
            np.hstack((objd_aug['img'], lbl_aug))
        ], (2, 1))
        grasp_fusion_lib.io.imshow(viz)
        if grasp_fusion_lib.io.waitkey() == ord('q'):
            break


def main():
    seg_dataset = grasp_fusion_lib.datasets.apc.JskARC2017DatasetV3(
        split='valid')
    object_data = grasp_fusion_lib.aug.seg_dataset_to_object_data(
        seg_dataset, random_state=np.random.RandomState(1),
        ignore_labels=[-1, 0, 41])
    object_data_aug = grasp_fusion_lib.aug.augment_object_data(
        object_data, random_state=np.random.RandomState(1))
    # create copy of the object_data
    object_data = grasp_fusion_lib.aug.seg_dataset_to_object_data(
        seg_dataset, random_state=np.random.RandomState(1),
        ignore_labels=[-1, 0, 41])
    view_object_data(object_data, object_data_aug, seg_dataset.class_names)


if __name__ == '__main__':
    main()
