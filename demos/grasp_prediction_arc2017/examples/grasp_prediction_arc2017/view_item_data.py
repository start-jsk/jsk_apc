#!/usr/bin/env python

import argparse
import copy
import os.path as osp

import numpy as np
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

    object_names, item_data = \
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

        def __init__(self, object_names, item_data, augment=False):
            self.split = 'all'
            self.class_names = ['__background__'] + object_names
            self.item_data = item_data
            self.augment = augment

        def __len__(self):
            return len(self.item_data)

        def __getitem__(self, i):
            if not self.augment:
                objd = self.item_data[i]
                return objd['img'], objd['lbl'], objd['lbl_suc']

            item_data = copy.deepcopy(self.item_data)
            random_state = np.random.RandomState(i)
            random_state.shuffle(item_data)

            img = np.zeros((500, 500, 3), dtype=np.uint8)
            img[:, :, 0] = 255  # red
            lbl = np.zeros(img.shape[:2], dtype=np.int32)
            stacked = mvtk.aug.stack_objects(
                img, lbl, item_data,
                region_label=0, random_state=random_state)
            img, lbl, lbl_suc = \
                stacked['img'], stacked['lbl'], stacked['lbl_suc']

            return img, lbl, lbl_suc

    dataset = ItemData(object_names, item_data, augment)

    def visualize_func(dataset, index):
        img, lbl, lbl_suc = dataset[index]
        lbl_viz = mvtk.datasets.visualize_label(
            lbl, img, class_names=dataset.class_names)
        lbl_suc_viz = mvtk.datasets.visualize_label(
            lbl_suc, img, class_names=['no_suction', 'suction'])
        return np.hstack((img, lbl_viz, lbl_suc_viz))

    mvtk.datasets.view_dataset(dataset, visualize_func)


if __name__ == '__main__':
    main()
