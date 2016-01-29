import os
import yaml

import rospkg

PKG = 'jsk_apc2015_common'
rp = rospkg.RosPack()


def get_object_list():
    pkg_path = rp.get_path(PKG)
    yaml_file = os.path.join(pkg_dir, 'data/object_list.yml')
    with open(yaml_file) as f:
        objects = yaml.load(f)
    return objects
