#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import yaml
import gzip
import cPickle as pickle

import cv2

import rospy
from jsk_2014_picking_challenge.srv import ObjectMatch, ObjectMatchResponse


def get_data_dir():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '../data')
    sub_data_dir = lambda x: os.path.join(data_dir, x)
    for sub in ['siftdata', 'histogram_data', 'bof_data']:
        if not os.path.exists(sub_data_dir(sub)):
            os.mkdir(sub_data_dir(sub))
    return data_dir


def load_img(imgpath):
    img = cv2.imread(imgpath)
    if img is None:
        rospy.logerr('not found {}'.format(imgpath))
    return img


def get_object_list():
    data_dir = get_data_dir()
    yaml_file = os.path.join(data_dir, 'object_list.yml')
    with open(yaml_file, 'rb') as f:
        return yaml.load(f)


def save_siftdata(obj_name, siftdata):
    """Save sift data to data/siftdata/{obj_name}.pkl.gz"""
    data_dir = get_data_dir()
    siftdata_dir = os.path.join(data_dir, 'siftdata')
    if not os.path.exists(siftdata_dir):
        os.mkdir(siftdata_dir)
    filename = os.path.join(siftdata_dir, obj_name+'.pkl.gz')
    rospy.loginfo('save siftdata: {o}'.format(o=obj_name))
    with gzip.open(filename, 'wb') as f:
        pickle.dump(siftdata, f)


def load_siftdata(obj_name, return_pos=True, dry_run=False):
    """Load sift data from pkl file"""
    data_dir = get_data_dir()
    datafile = os.path.join(data_dir, 'siftdata/{}.pkl.gz'.format(obj_name))
    if dry_run:  # check if exists
        if os.path.exists(datafile):
            return datafile
        else:
            return
    if not os.path.exists(datafile):
        rospy.logerr('not found siftdata: {}'.format(obj_name))
        return  # does not exists
    rospy.loginfo('load siftdata: {o}'.format(o=obj_name))
    with gzip.open(datafile, 'rb') as f:
        siftdata = pickle.load(f)
    if return_pos:
        return siftdata
    return siftdata['descriptors']


def get_train_imgpaths(obj_name):
    """Find train image paths from data/obj_name"""
    data_dir = get_data_dir()
    obj_dir = os.path.join(data_dir, obj_name)
    if not os.path.exists(obj_dir):
        rospy.logwarn('not found object data: {o}'.format(o=obj_name))
        return
    os.chdir(obj_dir)
    imgpaths = []
    for imgfile in os.listdir('.'):
        if not imgfile.endswith('.jpg'):
            continue
        raw_path = os.path.join(obj_dir, imgfile)
        maskfile = os.path.splitext(imgfile)[0] + '_mask.pbm'
        mask_path = os.path.join(obj_dir, 'masks', maskfile)
        imgpaths.append((raw_path, mask_path))
    os.chdir(data_dir)
    if len(imgpaths) == 0:
        rospy.logwarn('not found images: {o}'.format(o=obj_name))
        return
    return imgpaths


class ObjectMatcher(object):
    def __init__(self, service_name):
        rospy.Service(service_name, ObjectMatch, self._cb_matcher)

    def _cb_matcher(self, req):
        """Callback function for sift match request"""
        rospy.loginfo('received request: {}'.format(req.objects))
        probs = self.match(req.objects)
        return ObjectMatchResponse(probabilities=probs)

    def match(self, obj_names):
        """Get object match probabilities"""
        raise NotImplementedError('override this method')


def is_imgfile(filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.jpg', '.jpeg', '.png', '.pgm']:
        return True
    return False


def listdir_for_img(data_dir):
    for f in os.listdir(data_dir):
        if is_imgfile(f):
            yield f

