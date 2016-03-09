#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

import yaml
import rospkg
import json

import cv2
import numpy as np

import jsk_apc2015_common
import rospkg

rp = rospkg.RosPack()
PKG = 'jsk_apc2016_common'
PKG_PATH = rp.get_path(PKG)
OLD_PKG = 'jsk_apc2015_common'##Delete this path and change all OLD_PKG_PATH to PKG_PATH after new objects for apc2016 arrive from Amazon
OLD_PKG_PATH = rp.get_path(OLD_PKG)

def get_object_data():
    fname = osp.join(rp.get_path(PKG), 'data/object_data.yaml')
    data = yaml.load(open(fname))
    return data

def load_stow_json(json_file):
    json_data = json.load(open(json_file))
    bin_contents = {}
    for bin, item in json_data['bin_contents'].items():
        bin = bin[len('bin_'):].lower()
        bin_contents[bin] = item
    work_order = []
    for contents in json_data['work_order']:
        work_order.append(contents)
    return bin_contents, work_order

def visualize_stow_json(json_file):
    """Visualize bin_contents and work_order with a json file"""
    from jsk_apc2015_common.util import rescale
    # load data from json
    bin_contents, work_order = load_stow_json(json_file)
    # initialize variables
    kiva_pod_img = cv2.imread(osp.join(OLD_PKG_PATH, 'models/kiva_pod/image.jpg'))
    tote_img = cv2.imread(osp.join(PKG_PATH, 'models/tote/image.jpg'))

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
    object_list = get_object_data()
    object_imgs = {}
    for obj in object_list:
        obj = obj['name']
        img_path = osp.join(OLD_PKG_PATH, 'models/{obj}/image.jpg'.format(obj=obj))
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if h > w:
            img = np.rollaxis(img, 1)
        object_imgs[obj] = img
    # draw objects
    for bin, contents in bin_contents.items():
        bin_pt1, bin_pt2 = BIN_REGION[bin]
        bin_region = kiva_pod_img[bin_pt1[1]:bin_pt2[1], bin_pt1[0]:bin_pt2[0]]
        x_num, y_num = jsk_apc2015_common._get_tile_shape(len(contents))
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


    # draw bin regions
    for bin, region in BIN_REGION.items():
        bin_pt1, bin_pt2 = region
        bin_h, bin_w = bin_pt2[1] - bin_pt1[1], bin_pt2[0] - bin_pt1[0]
        cv2.rectangle(kiva_pod_img, bin_pt1, bin_pt2, (0, 0, 255), 2)
        text_pos = (bin_pt2[0] - bin_h//2 - 100, bin_pt2[1])
        cv2.putText(kiva_pod_img, bin.upper(), text_pos,
                    cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 3)
    

    # draw objects in tote
    tote_region = [[190,230],[1080,790]]
    region_h = tote_region[1][1] - tote_region[0][1]
    region_w = tote_region[1][0] - tote_region[0][0]
    max_obj_h,max_obj_w = region_h / 3, region_w / 4
    tote_x_min,tote_y_min = tote_region[0][0], tote_region[0][1]
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
        if x_max >= region_w :
            x_min = tote_x_min
            y_min += max_obj_h

    # merge two images
    kiva_w,kiva_h = kiva_pod_img.shape[1],kiva_pod_img.shape[0]
    tote_w,tote_h = tote_img.shape[1],tote_img.shape[0]
    tote_img = cv2.resize(tote_img,(kiva_w,tote_h*kiva_w//tote_w))
    dest = np.concatenate((kiva_pod_img,tote_img),axis=0)
    return dest
