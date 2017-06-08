#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
from collections import OrderedDict
import json
import os
import os.path as osp

import numpy as np
import skimage.io
import skimage.transform
import yaml

import rospkg


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')


def _patch_json_content(content):
    assert isinstance(content, unicode)
    lines = content.splitlines()
    lines[1] = lines[1].replace(u'”', '"')
    content = '\n'.join(lines)
    return content


def _patch_img_fname(img_fname):
    basedir, basename = osp.split(img_fname)
    if basename == 'Plastic_Wine_Glass_Top_01.png':
        return osp.join(basedir, 'Plastic_Wine _Glass_Top_01.png')
    elif basename == 'Tissue_Box_Top_01.png':
        return osp.join(basedir, 'Tissue_Box_top_01.png')
    return img_fname


def main():
    # TODO(unknown)
    # classes = []
    objects = []
    grasp_types = []

    data_dir = osp.join(PKG_DIR, 'data')
    dataset_dir = osp.join(data_dir, 'datasets/AR20170331')
    if not osp.exists(dataset_dir):
        url = 'https://drive.google.com/open?id=0B9P1L--7Wd2vSjI5a3hiMU04THc'
        print('Please download dataset from %s,\nand extract it to %s.' %
              (url, dataset_dir))
        quit(1)

    print('==> Parsing dataset: %s' % dataset_dir)
    for obj_dir in sorted(os.listdir(dataset_dir)):
        obj_name_capital = obj_dir
        obj_name = obj_name_capital.lower()
        obj_dir = osp.join(dataset_dir, obj_dir)

        out_dir = osp.join(PKG_DIR, 'config/objects', obj_name)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        print('==> Writing to: %s' % out_dir)

        # info.json
        json_file = osp.join(obj_dir, '%s.json' % obj_name)
        content = codecs.open(json_file, encoding='utf-8').read()
        content = _patch_json_content(content)
        obj_datum = json.loads(content, object_pairs_hook=OrderedDict)
        with open(osp.join(out_dir, 'info.json'), 'w') as f:
            f.write(json.dumps(obj_datum, sort_keys=False, indent=4) + '\n')
        objects.append(obj_datum['name'])
        grasp_types.append(obj_datum['type'])

        # top.jpg
        img_file = osp.join(obj_dir, '%s_Top_01.png' % obj_name_capital)
        img = skimage.io.imread(_patch_img_fname(img_file))
        # resize to 500 x 500
        img_sqr = np.zeros((500, 500, 3), dtype=np.float64)
        scale = 500. / max(img.shape[:2])
        img = skimage.transform.rescale(img, scale)
        img_sqr[:img.shape[0], :img.shape[1]] = img
        skimage.io.imsave(osp.join(out_dir, 'top.jpg'), img_sqr)

    out_dir = osp.join(PKG_DIR, 'config')
    print('==> Writing to: %s' % out_dir)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # TODO(unknown)
    # classes.txt
    with open(osp.join(out_dir, 'classes.txt'), 'w') as f:
        pass

    # label_names.yaml
    with open(osp.join(PKG_DIR, 'config/label_names.yaml'), 'w') as f:
        label_names = ['__background_'] + objects + ['__shelf__']
        yaml.dump(label_names, f)

    # grasp_types.txt
    with open(osp.join(out_dir, 'grasp_types.txt'), 'w') as f:
        grasp_types = sorted(list(set(grasp_types)))
        f.write('\n'.join(grasp_types))


if __name__ == '__main__':
    main()
