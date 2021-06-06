#!/usr/bin/env python

import os
import os.path as osp


filepath = osp.dirname(osp.realpath(__file__))
datadir = osp.join(filepath, '../data/objects')


def main():
    for objectname in os.listdir(datadir):
        objectdir = osp.join(datadir, objectname)
        if not osp.isdir(objectdir):
            continue
        for d in os.listdir(objectdir):
            imgpath = osp.join(objectdir, d, 'rgb.png')
            if not osp.exists(imgpath):
                continue
            jsonpath = osp.join(objectdir, d, 'grasping_point.json')
            print('imgpath: {}'.format(imgpath))
            os.system('labelme {0} -O {1}'.format(imgpath, jsonpath))

if __name__ == '__main__':
    main()
