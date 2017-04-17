#!/usr/bin/env python

import os
import os.path as osp

import numpy as np
import PIL.Image
import skimage.color

import jsk_apc2016_common
import jsk_recognition_utils
import rospkg

from dataset import get_object_names


PKG_PATH = rospkg.RosPack().get_path('jsk_arc2017_common')


def get_class_id_map():
    cls_names_16 = ['__background__']
    cls_names_16 += [d['name'] for d in jsk_apc2016_common.get_object_data()]

    cls_names_17 = get_object_names()

    cls_name_16_to_17 = {
        '__background__': '__shelf__',
        # 'womens_knit_gloves': 'black_fashion_gloves',
        'crayola_24_ct': 'crayons',
        'scotch_duct_tape': 'duct_tape',
        'expo_dry_erase_board_eraser': 'expo_eraser',
        'hanes_tube_socks': 'hanes_socks',
        'laugh_out_loud_joke_book': 'laugh_out_loud_jokes',
        'rolodex_jumbo_pencil_cup': 'mesh_cup',
        'ticonderoga_12_pencils': 'ticonderoga_pencils',
        'kleenex_tissue_box': 'tissue_box',
    }

    print('{:>28} -> {:<15}'.format('apc2016', 'arc2017'))
    print('-' * 53)
    cls_id_16_to_17 = {}
    for n16, n17 in cls_name_16_to_17.items():
        assert n16 in cls_names_16
        assert n17 in cls_names_17
        print('{:>28} -> {:<15}'.format(n16, n17))
        cls_id_16_to_17[cls_names_16.index(n16)] = cls_names_17.index(n17)

    return cls_id_16_to_17


def main():
    cls_id_16_to_17 = get_class_id_map()

    cmap = jsk_recognition_utils.color.labelcolormap(41)
    dataset_dir = osp.join(PKG_PATH, 'data/datasets/JSKAPC2016')
    out_dir = osp.join(PKG_PATH, 'data/datasets/JSKARC2017From16')
    for scene_dir in os.listdir(dataset_dir):
        out_sub_dir = osp.join(out_dir, osp.basename(scene_dir))

        scene_dir = osp.join(dataset_dir, scene_dir)
        img_file = osp.join(scene_dir, 'image.png')
        lbl_file = osp.join(scene_dir, 'label.png')
        img = np.array(PIL.Image.open(img_file))
        assert img.dtype == np.uint8
        lbl = np.array(PIL.Image.open(lbl_file), dtype=np.int32)
        lbl[lbl == 255] = -1
        for i16 in np.unique(lbl):
            if i16 in cls_id_16_to_17:
                lbl[lbl == i16] = cls_id_16_to_17[i16]
            else:
                lbl[lbl == i16] = -1

        if np.unique(lbl).tolist() == [-1, 41]:
            print('==> Skipping scene without 2017 objects: %s' % out_sub_dir)
            continue

        if not osp.exists(out_sub_dir):
            os.makedirs(out_sub_dir)

        lbl_viz = skimage.color.label2rgb(lbl, img, bg_label=0,
                                          colors=cmap[1:])
        lbl_viz = (lbl_viz * 255).astype(np.uint8)
        lbl_viz[lbl == -1] = (0, 0, 0)

        out_img_file = osp.join(out_sub_dir, 'image.jpg')
        PIL.Image.fromarray(img).save(out_img_file)
        out_lbl_file = osp.join(out_sub_dir, 'label.npz')
        np.savez_compressed(out_lbl_file, lbl)
        out_lbl_viz_file = osp.join(out_sub_dir, 'label_viz.jpg')
        PIL.Image.fromarray(lbl_viz).save(out_lbl_viz_file)

        print('==> Saved to: %s' % out_sub_dir)


if __name__ == '__main__':
    main()
