#!/usr/bin/env python

import argparse
import cv2
import os
import os.path as osp
import zipfile


filepath = osp.dirname(osp.realpath(__file__))
datadir = osp.join(filepath, '../data/objects')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zipfile', '-z', default=None)
    args = parser.parse_args()
    zippath = args.zipfile
    if zippath is None:
        zippath = osp.join(filepath, '../data/compressed/ItemDataARC2017.zip')

    with zipfile.ZipFile(zippath, 'r') as zipf:
        zipf.extractall('/tmp')

    extractdir = '/tmp/ItemDataARC2017'
    objectnames = os.listdir(extractdir)

    for objectname in objectnames:
        objectdir = osp.join(datadir, objectname.lower())
        print('object: {}'.format(objectname.lower()))
        if not osp.exists(objectdir):
            os.makedirs(objectdir)
        pngnames = os.listdir(osp.join(extractdir, objectname))
        pngnames = [f for f in pngnames if f.endswith('.png')]
        for i, pngname in enumerate(pngnames):
            pngpath = osp.join(extractdir, objectname, pngname)
            dst_dir = osp.join(objectdir, '{0:02d}'.format(i))
            if not osp.exists(dst_dir):
                os.mkdir(dst_dir)
            img = cv2.imread(pngpath)
            img = cv2.resize(img, (500, 500))
            dst_pngpath = osp.join(dst_dir, 'rgb.png')
            cv2.imwrite(dst_pngpath, img)


if __name__ == '__main__':
    main()
