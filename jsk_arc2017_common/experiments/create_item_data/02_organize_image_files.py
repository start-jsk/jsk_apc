#!/usr/bin/env python

import json
import os
import os.path as osp

import skimage.io


here = osp.dirname(osp.abspath(__file__))


def main():
    item_data_dir = osp.join(here, 'item_data')

    for item_name_upper in os.listdir(item_data_dir):
        item_dir = osp.join(item_data_dir, item_name_upper)
        if not osp.isdir(item_dir):
            continue

        item_name_lower = item_name_upper.lower()

        frame_id = 0
        for fname in sorted(os.listdir(item_dir)):
            if fname.startswith(item_name_upper):
                continue
            fname = osp.join(item_dir, fname)
            ext = osp.splitext(fname)[1]
            if not (osp.isfile(fname) and ext in ['.jpg', '.png']):
                continue
            img = skimage.io.imread(fname)
            frame_id += 1
            dst_fname = osp.join(
                item_dir, '{:s}_{:03d}.png'.format(item_name_upper, frame_id))
            skimage.io.imsave(dst_fname, img)
            print('{:s}: {:s} -> {:s}'
                  .format(item_name_lower, fname, dst_fname))
            os.remove(fname)


if __name__ == '__main__':
    main()
