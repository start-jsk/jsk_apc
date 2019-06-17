#!/usr/bin/env python

import glob
import json
import os.path as osp
import subprocess

import six
import skimage.io

import contrib


def main():
    data = None
    for img_file in sorted(glob.glob('*.jpg')):
        out_file = osp.splitext(img_file)[0] + '.json'
        print('Annotation: %s -> %s' % (img_file, out_file))
        if osp.exists(out_file):
            data = json.load(open(out_file))
            continue

        while True:
            yn = six.moves.input('Going %s? [yn]: ' % img_file).strip().lower()
            if yn == 'y':
                break
            elif yn == 'n':
                return

        if data is not None:
            data['imagePath'] = img_file
            img = skimage.io.imread(img_file)
            data['imageData'] = contrib.utils.ndarray_to_base64(img)
            json.dump(data, open(out_file, 'w'))
            cmd = 'labelme %s' % out_file
        else:
            cmd = 'labelme %s -O %s' % (img_file, out_file)

        print('+ %s' % cmd)
        subprocess.call(cmd, shell=True)

        data = json.load(open(out_file))


if __name__ == '__main__':
    main()
