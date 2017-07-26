import os.path as osp

import skimage.color
import yaml

import rospkg


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')


def get_object_weights():
    with open(osp.join(PKG_DIR, 'config/object_weights.yaml')) as f:
        return yaml.load(f)


def get_label_names():
    with open(osp.join(PKG_DIR, 'config/label_names.yaml')) as f:
        return yaml.load(f)


def get_object_names():
    label_names = get_label_names()
    return [l for l in label_names if not l.startswith('__')]


def get_known_object_names():
    with open(osp.join(PKG_DIR, 'data/names/known_object_names.yaml')) as f:
        return yaml.load(f)


def get_object_images():
    object_names = get_object_names()
    object_imgs = {}
    for obj in object_names:
        obj_file = osp.join(PKG_DIR, 'data/objects', obj, 'top.jpg')
        img_obj = skimage.io.imread(obj_file)
        object_imgs[obj] = img_obj
    return object_imgs


def get_object_graspability():
    with open(osp.join(PKG_DIR, 'config/object_graspability.yaml')) as f:
        return yaml.load(f)
