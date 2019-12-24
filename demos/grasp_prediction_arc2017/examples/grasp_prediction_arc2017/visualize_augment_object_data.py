#!/usr/bin/env python

import os
import os.path as osp

import numpy as np
import skimage.io
import tqdm

import mvtk


def visualize_func(objd, objd_aug):
    img, lbl, lbl_suc = [objd[k] for k in ['img', 'lbl', 'lbl_suc']]
    lbl_viz = mvtk.datasets.visualize_label(lbl, img)
    lbl_suc_viz = mvtk.datasets.visualize_label(lbl_suc, img)
    viz1 = np.hstack((img, lbl_viz, lbl_suc_viz))

    img, lbl, lbl_suc = [objd_aug[k] for k in ['img', 'lbl', 'lbl_suc']]
    lbl_viz = mvtk.datasets.visualize_label(lbl, img)
    lbl_suc_viz = mvtk.datasets.visualize_label(lbl_suc, img)
    viz2 = np.hstack((img, lbl_viz, lbl_suc_viz))

    return mvtk.image.tile([viz1, viz2], shape=(2, 1))


def main():
    random_state = np.random.RandomState(1)

    item_data_dir = osp.expanduser('~/data/arc2017/item_data/pick_re-experiment')  # NOQA
    obj_names, obj_data = mvtk.datasets.apc.arc2017.load_item_data(
        item_data_dir)
    for i, objd in tqdm.tqdm(enumerate(obj_data)):
        objd_aug = mvtk.aug.augment_object_data(
            [objd.copy()], random_state).next()
        objd_aug['lbl'] = mvtk.image.mask_to_lbl(
            objd_aug['mask'], objd_aug['label'])
        viz = visualize_func(objd, objd_aug)
        obj_name = obj_names[objd['label'] - 1]
        img_file = osp.join('logs/out_visualize_augment_object_data',
                            obj_name, '%06d.jpg' % i)
        if not osp.exists(osp.dirname(img_file)):
            os.makedirs(osp.dirname(img_file))
        skimage.io.imsave(img_file, viz)
        # mvtk.io.imshow(viz)
        # if mvtk.io.waitkey(0) == ord('q'):
        #     break


if __name__ == '__main__':
    main()
