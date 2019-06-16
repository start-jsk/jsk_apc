import os.path as osp

import yaml


here = osp.dirname(osp.abspath(__file__))

with open(osp.join(here, 'data/object_names.yaml')) as f:
    class_names_apc2016 = yaml.safe_load(f)
