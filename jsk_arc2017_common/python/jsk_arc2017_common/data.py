import os.path as osp

import skimage.color
import yaml

import rospkg


def get_object_weights():
    PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')

    with open(osp.join(PKG_DIR, 'config/object_weights.yaml')) as f:
        return yaml.load(f)


def get_object_names():
    PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')

    with open(osp.join(PKG_DIR, 'config/label_names.yaml')) as f:
        return yaml.load(f)


def get_object_images():
    object_names = get_object_names()[1:-1]
    object_imgs = {}
    PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')
    for obj in object_names:
        obj_file = osp.join(PKG_DIR, 'data/objects', obj, 'top.jpg')
        img_obj = skimage.io.imread(obj_file)
        object_imgs[obj] = img_obj
    return object_imgs


def get_object_graspability():
    PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')

    with open(osp.join(PKG_DIR, 'config/object_graspability.yaml')) as f:
        return yaml.load(f)
