import json
import math
import os.path as osp
import yaml

import cv2
import numpy as np

import rospkg


PKG = 'jsk_apc2015_common'


def get_object_list():
    """Returns the object name list for APC2015.

    Args:
        None.

    Returns:
        objects (list): List of object name.
    """
    pkg_path = rospkg.RosPack().get_path(PKG)
    yaml_file = osp.join(pkg_path, 'data/object_list.yml')
    with open(yaml_file) as f:
        objects = yaml.load(f)
    return objects


def load_json(json_file):
    """Load the json file which is interface for APC2015.

    Args:
        json_file (str): Path to the json file.
    """
    json_data = json.load(open(json_file))
    bin_contents = {}
    for bin, contents in json_data['bin_contents'].items():
        bin = bin[len('bin_'):].lower()
        bin_contents[bin] = contents
    work_order = {}
    for order in json_data['work_order']:
        bin = order['bin'][len('bin_'):].lower()
        work_order[bin] = order['item']
    return bin_contents, work_order


def _get_tile_shape(img_num):
    x_num = 0
    y_num = int(round((math.sqrt(img_num))))
    while x_num * y_num < img_num:
        x_num += 1
    return x_num, y_num


def visualize_bin_contents(bin_contents, work_order=None,
                           extra_img_paths=None):
    """Returns visualized image of bin contents.

    Args:
        bin_contents (dict): contents of each bin.
        work_order (dict): target objects for each bin (default: ``None``).
        extra_img_paths (dict): {object_name: img_path}

    Returns:
        kiva_pod_img (~numpy.ndarray):
            visualized image of listed objects over the Kiva Pod image.
    """
    from jsk_apc2015_common.util import rescale
    pkg_path = rospkg.RosPack().get_path(PKG)
    object_img_paths = {
        obj: osp.join(pkg_path, 'models/{0}/image.jpg'.format(obj))
        for obj in get_object_list()
    }
    if extra_img_paths is not None:
        object_img_paths.update(extra_img_paths)
    # initialize variables
    kiva_pod_img = cv2.imread(osp.join(pkg_path, 'models/kiva_pod/image.jpg'))
    BIN_REGION = {
        'a': ((0, 50), (640, 610)),
        'b': ((640, 50), (1410, 610)),
        'c': ((1410, 50), (2060, 610)),
        'd': ((0, 680), (640, 1200)),
        'e': ((640, 680), (1410, 1200)),
        'f': ((1410, 680), (2060, 1200)),
        'g': ((0, 1280), (640, 1770)),
        'h': ((640, 1280), (1410, 1770)),
        'i': ((1410, 1280), (2060, 1770)),
        'j': ((0, 1850), (640, 2430)),
        'k': ((640, 1850), (1410, 2430)),
        'l': ((1410, 1850), (2060, 2430)),
    }
    # get object images
    object_imgs = {}
    for obj, img_path in object_img_paths.items():
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if h > w:
            img = np.rollaxis(img, 1)
        object_imgs[obj] = img
    # draw objects
    for bin, contents in bin_contents.items():
        if not contents:
            continue  # skip empty bin
        bin_pt1, bin_pt2 = BIN_REGION[bin]
        bin_region = kiva_pod_img[bin_pt1[1]:bin_pt2[1], bin_pt1[0]:bin_pt2[0]]
        x_num, y_num = _get_tile_shape(len(contents))
        bin_h, bin_w = bin_region.shape[:2]
        max_obj_h, max_obj_w = bin_h // y_num, bin_w // x_num
        for i_y in xrange(y_num):
            y_min = int(1. * bin_h / y_num * i_y)
            for i_x in xrange(x_num):
                x_min = int(1. * bin_w / x_num * i_x)
                if contents:
                    obj = contents.pop()
                    obj_img = object_imgs[obj]
                    scale_h = 1. * max_obj_h / obj_img.shape[0]
                    scale_w = 1. * max_obj_w / obj_img.shape[1]
                    scale = min([scale_h, scale_w])
                    obj_img = rescale(obj_img, scale)
                    obj_h, obj_w = obj_img.shape[:2]
                    x_max, y_max = x_min + obj_w, y_min + obj_h
                    bin_region[y_min:y_max, x_min:x_max] = obj_img
                    # highlight work order
                    if work_order and work_order[bin] == obj:
                        pt1 = (x_min + 10, y_min + 10)
                        pt2 = (x_max - 10, y_max - 10)
                        cv2.rectangle(bin_region, pt1, pt2, (0, 255, 0), 8)
    # draw bin regions
    for bin, region in BIN_REGION.items():
        bin_pt1, bin_pt2 = region
        bin_h, bin_w = bin_pt2[1] - bin_pt1[1], bin_pt2[0] - bin_pt1[0]
        cv2.rectangle(kiva_pod_img, bin_pt1, bin_pt2, (0, 0, 255), 2)
        text_pos = (bin_pt2[0] - bin_h//2 - 100, bin_pt2[1])
        cv2.putText(kiva_pod_img, bin.upper(), text_pos,
                    cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 6)
    return kiva_pod_img


def visualize_json(json_file):
    """Visualize bin_contents and work_order with a json file

    Args:
        json_file (``str``): Path to the json file.

    Returns:
        kiva_pod_img (~numpy.ndarray):
            visualized image of listed objects over the Kiva Pod image.
    """
    # load data from json
    bin_contents, work_order = load_json(json_file)
    return visualize_bin_contents(bin_contents, work_order)
