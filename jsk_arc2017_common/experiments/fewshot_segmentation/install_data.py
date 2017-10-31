#!/usr/bin/env python

import functools
import os
import os.path as osp
import shutil

import yaml

import mvtk


def main():
    # ~/data/arc2017/datasets/ItemDataAll
    dataset_dir = osp.expanduser('~/data/arc2017/datasets')
    url = 'https://www.dropbox.com/s/j8cka6mmna80gn2/ItemDataAll.zip?dl=1'
    path = osp.join(dataset_dir, 'ItemDataAll.zip')
    postprocess = functools.partial(
        mvtk.data.extract, path=path, to=dataset_dir)
    mvtk.data.download_mp(url=url, path=path, postprocess=postprocess)

    # ~/data/arc2017/item_data
    item_data_all_dir = osp.join(dataset_dir, 'ItemDataAll')
    here = osp.dirname(osp.realpath(__file__))
    yaml_dir = osp.join(here, 'item_data_yaml')
    for yaml_file in os.listdir(yaml_dir):
        item_data_name = osp.splitext(yaml_file)[0]

        yaml_file = osp.join(yaml_dir, yaml_file)
        objects = yaml.load(open(yaml_file))

        item_data_dir = osp.join(
            osp.expanduser('~/data/arc2017/item_data'), item_data_name)
        if osp.exists(item_data_dir):
            print('ItemData exists: %s' % item_data_dir)
            continue

        os.makedirs(item_data_dir)
        for obj in objects:
            shutil.copytree(
                osp.join(item_data_all_dir, obj),
                osp.join(item_data_dir, obj),
            )
        print('Created ItemData: %s' % item_data_dir)

    # ~/data/arc2017/models/fcn32s_cfg012_arc2017_iter00140000_20170729.npz
    url = 'https://www.dropbox.com/s/8dmifq2tusrmwpf/fcn32s_cfg012_arc2017_iter00140000_20170729.npz?dl=1'
    path = osp.expanduser('~/data/arc2017/models/fcn32s_cfg012_arc2017_iter00140000_20170729.npz')
    mvtk.data.download_mp(url=url, path=path)


if __name__ == '__main__':
    main()
