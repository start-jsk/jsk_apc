#!/usr/bin/env python

import argparse
import os
import os.path as osp
import scipy.misc

import rospkg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('-d', '--dir', default='v5')
    args = parser.parse_args()

    # answer candidates
    # default is no
    yes = ['yes', 'y']
    no = ['no', 'n', '']

    rospack = rospkg.RosPack()
    datasetdir = osp.join(
        rospack.get_path('selective_dualarm_stowing'), 'dataset', args.dir)
    for i, dirname in enumerate(os.listdir(datasetdir)):
        datadir = osp.join(datasetdir, dirname, 'after_stow')
        print('count: {}'.format(i+1))
        print(datadir)
        afterdirs = os.listdir(datadir)
        afterdirs = [x for x in afterdirs if osp.isdir(osp.join(datadir, x))]
        assert len(afterdirs) == 1
        afterdir = osp.join(datadir, afterdirs[0])

        if not args.overwrite and osp.exists(osp.join(afterdir, 'label.txt')):
            continue

        # is_bimanual
        with open(osp.join(afterdir, 'is_bimanual.txt'), 'r') as f:
            data = f.read()
            if data == 'True':
                is_bimanual = True
            else:
                is_bimanual = False
        if is_bimanual:
            style = 'dualarm'
        else:
            style = 'singlearm'

        imgpath = osp.join(afterdir, 'default_camera_image.png')
        img = scipy.misc.imread(imgpath)
        scipy.misc.imshow(img)

        # ask dropped
        while True:
            print('Dropped? [y/N]')
            choice = raw_input().lower()
            if choice in yes:
                dropped = True
                break
            elif choice in no:
                dropped = False
                break
            else:
                print('Type yes/no')

        # ask protruded
        while True:
            print('protruded? [y/N]')
            choice = raw_input().lower()
            if choice in yes:
                protruded = True
                break
            elif choice in no:
                protruded = False
                break
            else:
                print('Type yes/no')

        # ask damaged
        while True:
            print('damaged? [y/N]')
            choice = raw_input().lower()
            if choice in yes:
                damaged = True
                break
            elif choice in no:
                damaged = False
                break
            else:
                print('Type yes/no')

        label = []
        if is_bimanual:
            style = 'dualarm'
        else:
            style = 'singlearm'

        if not dropped and not protruded:
            label.append('success')
        else:
            if dropped:
                label.append('drop')
            if protruded:
                label.append('protrude')
            if damaged:
                label.append('damage')

        label = ['{0}_{1}'.format(style, x) for x in label]
        labeltxt = '\n'.join(label)

        labelpath = osp.join(afterdir, 'label.txt')
        with open(labelpath, 'w+') as f:
            f.write(labeltxt)


if __name__ == '__main__':
    main()
