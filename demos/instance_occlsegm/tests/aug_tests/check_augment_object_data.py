import numpy as np
import six

import instance_occlsegm_lib


def view_object_data(object_data, object_data_aug, class_names):
    for objd, objd_aug in six.moves.zip(object_data, object_data_aug):
        lbl = instance_occlsegm_lib.image.mask_to_lbl(
            objd['mask'], objd['label'])
        lbl_aug = instance_occlsegm_lib.image.mask_to_lbl(
            objd_aug['mask'], objd_aug['label'])

        objd['img'] = instance_occlsegm_lib.image.resize(
            objd['img'], size=200 * 200)
        lbl = instance_occlsegm_lib.image.resize(
            lbl, size=200 * 200, interpolation=0)
        objd_aug['img'] = instance_occlsegm_lib.image.resize(
            objd_aug['img'], size=200 * 200)
        lbl_aug = instance_occlsegm_lib.image.resize(
            lbl_aug, size=200 * 200, interpolation=0)

        lbl = instance_occlsegm_lib.datasets.visualize_label(
            lbl, objd['img'], class_names)
        lbl_aug = instance_occlsegm_lib.datasets.visualize_label(
            lbl_aug, objd_aug['img'], class_names)
        viz = instance_occlsegm_lib.image.tile([
            np.hstack((objd['img'], lbl)),
            np.hstack((objd_aug['img'], lbl_aug))
        ], (2, 1))
        instance_occlsegm_lib.io.imshow(viz)
        if instance_occlsegm_lib.io.waitkey() == ord('q'):
            break


def main():
    seg_dataset = instance_occlsegm_lib.datasets.apc.JskARC2017DatasetV3(
        split='valid')
    object_data = instance_occlsegm_lib.aug.seg_dataset_to_object_data(
        seg_dataset, random_state=np.random.RandomState(1),
        ignore_labels=[-1, 0, 41])
    object_data_aug = instance_occlsegm_lib.aug.augment_object_data(
        object_data, random_state=np.random.RandomState(1))
    # create copy of the object_data
    object_data = instance_occlsegm_lib.aug.seg_dataset_to_object_data(
        seg_dataset, random_state=np.random.RandomState(1),
        ignore_labels=[-1, 0, 41])
    view_object_data(object_data, object_data_aug, seg_dataset.class_names)


if __name__ == '__main__':
    main()
