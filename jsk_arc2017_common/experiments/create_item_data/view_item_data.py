#!/usr/bin/env python

import argparse
import os.path as osp

import termcolor

import mvtk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('item_data_dir')
    parser.add_argument('-a', '--augment', action='store_true')
    args = parser.parse_args()

    augment = args.augment

    item_data_dir = args.item_data_dir
    if not osp.exists(item_data_dir):
        print('Item data dir does not exist: %s' % item_data_dir)
        quit(1)

    if augment:
        dataset = mvtk.datasets.apc.arc2017.jsk.ItemDataDataset(
            split='train', item_data_dir=item_data_dir)
    else:
        object_names, imgs_and_lbls = \
            mvtk.datasets.apc.arc2017.load_item_data(item_data_dir)

        print('00: __background__')
        for i, obj_name in enumerate(object_names):
            obj_id = i + 1
            msg = '{:02}: {}'.format(obj_id, obj_name)
            if obj_name not in mvtk.datasets.apc.class_names_arc2017:
                termcolor.cprint(msg, color='red')
            else:
                print(msg)

        class ItemData(object):

            def __init__(self, object_names, imgs_and_lbls):
                self.split = 'all'
                self.class_names = ['__background__'] + object_names
                self.imgs_and_lbls = imgs_and_lbls

            def __len__(self):
                return len(self.imgs_and_lbls)

            def __getitem__(self, i):
                img, lbl = self.imgs_and_lbls[i]
                # aug = mvtk.lblaug.LabelAugmenter(
                #    object_labels=range(1, 33), region_label=0)
                # aug_img, _, _ = aug.get_augmenters(fit_output=True, scale=1)
                # img = aug_img.augment_image(img)
                return img, lbl

        dataset = ItemData(object_names, imgs_and_lbls)
    mvtk.datasets.view_class_seg_dataset(dataset)


if __name__ == '__main__':
    main()
