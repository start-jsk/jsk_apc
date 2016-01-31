import json
import os.path as osp
import yaml

import cv2
import numpy as np

import rospkg


PKG = 'jsk_apc2015_common'
rp = rospkg.RosPack()
PKG_PATH = rp.get_path(PKG)


def get_object_list():
    yaml_file = osp.join(PKG_PATH, 'data/object_list.yml')
    with open(yaml_file) as f:
        objects = yaml.load(f)
    return objects


def load_json(json_file):
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


def visualize_json(json_file):
    """Visualize bin_contents and work_order with a json file"""
    from jsk_apc2015_common.util import rescale
    # load data from json
    bin_contents, work_order = load_json(json_file)
    # initialize variables
    kiva_pod_img = cv2.imread(osp.join(PKG_PATH, 'models/kiva_pod/image.jpg'))
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
    object_list = get_object_list()
    object_imgs = {}
    for obj in object_list:
        img_path = osp.join(PKG_PATH, 'models/{obj}/image.jpg'.format(obj=obj))
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if h > w:
            img = np.rollaxis(img, 1)
        object_imgs[obj] = img
    # draw objects
    for bin, contents in bin_contents.items():
        bin_pt1, bin_pt2 = BIN_REGION[bin]
        bin_h, bin_w = bin_pt2[1] - bin_pt1[1], bin_pt2[0] - bin_pt1[0]
        for i, obj in enumerate(contents):
            if i == 0:
                pt1 = bin_pt1
            elif i == 1:
                pt1 = (bin_pt1[0] + bin_w // 2, bin_pt1[1])
            elif i == 2:
                pt1 = (bin_pt1[0], bin_pt1[1] + bin_h // 2)
            elif i == 3:
                pt1 = (bin_pt1[0] + bin_w // 2, bin_pt1[1] + bin_h // 2)
            max_obj_h, max_obj_w = bin_h // 2, bin_w // 2
            obj_img = object_imgs[obj]
            scale_h = 1. * max_obj_h / obj_img.shape[0]
            scale_w = 1. * max_obj_w / obj_img.shape[1]
            scale = min([scale_h, scale_w])
            obj_img = rescale(obj_img, scale)
            obj_h, obj_w = obj_img.shape[:2]
            pt2 = (pt1[0] + obj_w, pt1[1] + obj_h)
            kiva_pod_img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = obj_img
            # highlight work order
            if work_order[bin] == obj:
                pt1 = (pt1[0] + 10, pt1[1] + 10)
                pt2 = (pt2[0] - 10, pt2[1] - 10)
                cv2.rectangle(kiva_pod_img, pt1, pt2, (0, 255, 0), 3)
    # draw bin regions
    for bin, region in BIN_REGION.items():
        bin_pt1, bin_pt2 = region
        bin_h, bin_w = bin_pt2[1] - bin_pt1[1], bin_pt2[0] - bin_pt1[0]
        cv2.rectangle(kiva_pod_img, bin_pt1, bin_pt2, (0, 0, 255), 2)
        text_pos = (bin_pt2[0] - bin_h//2 - 100, bin_pt2[1])
        cv2.putText(kiva_pod_img, bin.upper(), text_pos,
                    cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 3)
    return kiva_pod_img
