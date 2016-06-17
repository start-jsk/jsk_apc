#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

import yaml
import rospkg
import json
import math

import cv2
import numpy as np

import jsk_apc2015_common
import rospkg


PKG = 'jsk_apc2016_common'


def get_object_list():
    """Returns the object name list for APC2016.

    Args:
        None.

    Returns:
        objects (list): List of object name.
    """
    pkg_path = rospkg.RosPack().get_path(PKG)
    yaml_file = osp.join(pkg_path, 'data/object_list.yaml')
    with open(yaml_file) as f:
        objects = yaml.load(f)
    return objects


def get_object_data():
    """Returns object data for APC2016.

    Returns:
        data (dict): objects data wrote in object_data.yaml file.
    """
    rp = rospkg.RosPack()
    fname = osp.join(rp.get_path(PKG), 'data/object_data_2016.yaml')
    data = yaml.load(open(fname))
    return data


def get_object_data_2015():
    """Returns object data for APC2015.

    Returns:
        data (dict): objects data wrote in object_data.yaml file.
    """
    rp = rospkg.RosPack()
    fname = osp.join(rp.get_path(PKG), 'data/object_data.yaml')
    data = yaml.load(open(fname))
    return data


def get_bin_contents(json_file):
    """Return bin contents data from picking json.

    Returns:
        data (dict): bin contents data written in picking json file.
    """
    with open(json_file, 'r') as f:
        bin_contents = json.load(f)['bin_contents']
    dict_contents = {}
    for bin_, objects in bin_contents.items():
        bin_ = bin_.split('_')[1].lower()  # bin_A -> a
        dict_contents[bin_] = objects
    return dict_contents


def get_work_order(json_file):
    """Return work order data from picking json.

    Returns:
        data (dict): work order written in picking json file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)['work_order']
    dict_order = {}
    for order in data:
        bin_ = order['bin'].split('_')[1].lower()  # bin_A -> a
        target_object = order['item']
        dict_order[bin_] = target_object
    return dict_order


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
    pkg_path_2015 = rospkg.RosPack().get_path('jsk_apc2015_common')
    kiva_pod_img = cv2.imread(osp.join(pkg_path_2015, 'models/kiva_pod/image.jpg'))
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
                        cv2.rectangle(bin_region, pt1, pt2, (0, 255, 0), 3)
    # draw bin regions
    for bin, region in BIN_REGION.items():
        bin_pt1, bin_pt2 = region
        bin_h, bin_w = bin_pt2[1] - bin_pt1[1], bin_pt2[0] - bin_pt1[0]
        cv2.rectangle(kiva_pod_img, bin_pt1, bin_pt2, (0, 0, 255), 2)
        text_pos = (bin_pt2[0] - bin_h//2 - 100, bin_pt2[1])
        cv2.putText(kiva_pod_img, bin.upper(), text_pos,
                    cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 3)
    return kiva_pod_img


def visualize_stow_contents(work_order):
    """Visualize stow contents with passed work order.

    Args:
        work_order (list): objects in the stow.

    Returns:
        tote_img (~numpy.ndarray): image of objects over the tote.
    """
    from jsk_apc2015_common.util import rescale
    rp = rospkg.RosPack()
    pkg_path = rp.get_path(PKG)
    tote_img = cv2.imread(osp.join(pkg_path, 'models/tote/image.jpg'))
    object_list = get_object_list()
    object_imgs = {}
    for obj in object_list:
        img_path = osp.join(pkg_path, 'models/{obj}/image.jpg'.format(obj=obj))
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if h > w:
            img = np.rollaxis(img, 1)
        object_imgs[obj] = img
    # draw object images on tote image
    tote_region = [[190, 230], [1080, 790]]
    region_h = tote_region[1][1] - tote_region[0][1]
    region_w = tote_region[1][0] - tote_region[0][0]
    max_obj_h, max_obj_w = region_h / 3, region_w / 4
    tote_x_min, tote_y_min = tote_region[0][0], tote_region[0][1]
    x_min, y_min = tote_x_min, tote_y_min
    for obj in work_order:
        obj_img = object_imgs[obj]
        scale_h = 1. * max_obj_h / obj_img.shape[0]
        scale_w = 1. * max_obj_w / obj_img.shape[1]
        scale = min([scale_h, scale_w])
        obj_img = rescale(obj_img, scale)
        obj_h, obj_w = obj_img.shape[:2]
        x_max, y_max = x_min + obj_w, y_min + obj_h
        tote_img[y_min:y_max, x_min:x_max] = obj_img
        x_min += max_obj_w
        if x_max >= region_w:
            x_min = tote_x_min
            y_min += max_obj_h
    return tote_img


def _load_stow_json(json_file):
    json_data = json.load(open(json_file))
    bin_contents = {}
    for bin, item in json_data['bin_contents'].items():
        bin = bin[len('bin_'):].lower()
        bin_contents[bin] = item
    tote_contents = []
    for contents in json_data['tote_contents']:
        tote_contents.append(contents)
    return bin_contents, tote_contents


def visualize_stow_json(json_file):
    """Visualize json file for Stow Task in APC2016.

    Args:
        json_file (str): Path to the json file.

    Returns:
        dest (~numpy.ndarray): Image of objects in bins and tote.
    """
    bin_contents, tote_contents = _load_stow_json(json_file)
    # draw bin contents
    kiva_pod_img = visualize_bin_contents(bin_contents)
    # draw tote contents
    tote_img = visualize_stow_contents(tote_contents)

    # merge two images
    kiva_w = kiva_pod_img.shape[1]
    tote_w, tote_h = tote_img.shape[1], tote_img.shape[0]
    tote_img = cv2.resize(tote_img, (kiva_w, tote_h*kiva_w//tote_w))
    dest = np.concatenate((kiva_pod_img, tote_img), axis=0)
    return dest


def visualize_pick_json(json_file):
    """Visualize json file for Pick Task in APC2016

    Args:
        json_file (``str``): Path to the json file.

    Returns:
        kiva_pod_img (~numpy.ndarray):
            visualized image of listed objects over the Kiva Pod image.
    """
    # load data from json
    bin_contents, work_order = jsk_apc2015_common.load_json(json_file)
    # set extra image paths that is added in APC2016
    rp = rospkg.RosPack()
    pkg_path = rp.get_path(PKG)
    extra_img_paths = {}
    for entry in get_object_data():
        obj = entry['name']
        extra_img_paths[obj] = osp.join(pkg_path, 'models', obj, 'image.jpg')
    # generate visualized image
    img = jsk_apc2015_common.visualize_bin_contents(
        bin_contents, work_order, extra_img_paths)
    return img
