import os.path as osp

import rospkg
import yaml


def get_object_names():
    PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')

    with open(osp.join(PKG_DIR, 'config/label_names.yaml')) as f:
        return yaml.load(f)
