#!/usr/bin/env python

import gdown
import os
import os.path as osp
import rospkg
import sys


if __name__ == '__main__':
    rospack = rospkg.RosPack()
    model_dir = osp.join(
        rospack.get_path('selective_dualarm_stowing'), 'models')
    chainermodel_path = osp.join(
        model_dir, 'bvlc_alexnet.chainermodel')
    if osp.exists(chainermodel_path):
        print('it is already downloaded')
        sys.exit(0)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    gdown.download(
        'https://drive.google.com/open?id=0B5DV6gwLHtyJZkd1ZTRiNUdrUXM',
        chainermodel_path,
        False)
