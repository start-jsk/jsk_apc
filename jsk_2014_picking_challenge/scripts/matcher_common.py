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


def load_img(imgpath):
    img = cv2.imread(imgpath)
    if img is None:
        rospy.logerr('Not found {}'.format(imgpath))
    return img


def get_object_list():
    dirname = os.path.dirname(os.path.abspath(__file__))
    ymlfile = os.path.join(dirname, '../data/object_list.yml')
    with open(ymlfile, 'rb') as f:
        return yaml.load(f)


def save_siftdata(obj_name, siftdata):
    """Save sift data to data/siftdata/{obj_name}.pkl.gz"""
    dirname = os.path.dirname(os.path.abspath(__file__))
    siftdata_dir = os.path.join(dirname, '../data/siftdata')
    if not os.path.exists(siftdata_dir):
        os.mkdir(siftdata_dir)
    filename = os.path.join(siftdata_dir, obj_name+'.pkl.gz')
    rospy.loginfo('Saving siftdata: {o}'.format(o=obj_name))
    with gzip.open(filename, 'wb') as f:
        pickle.dump(siftdata, f)


def load_siftdata(obj_name, dry_run=False):
    """Load sift data from pkl file"""
    dirname = os.path.dirname(os.path.abspath(__file__))
    datafile = os.path.join(dirname, '../data/siftdata',
                            obj_name+'.pkl.gz')
    if dry_run:  # check if exists
        if os.path.exists(datafile):
            return datafile
        else:
            return
    if not os.path.exists(datafile):
        rospy.logerr('not found siftdata: {}'.format(obj_name))
        return  # does not exists
    rospy.loginfo('loading siftdata: {o}'.format(o=obj_name))
    with gzip.open(datafile, 'rb') as f:
        return pickle.load(f)


def get_train_imgpaths(obj_name):
    """Find train image paths from data/obj_name"""
    dirname = os.path.dirname(os.path.abspath(__file__))
    obj_dir = os.path.join(dirname, '../data/', obj_name)
    if not os.path.exists(obj_dir):
        rospy.logwarn('Object data does not exists: {o}'.format(o=obj_name))
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
    os.chdir(dirname)
    if len(imgpaths) == 0:
        rospy.logwarn('Not found image files: {o}'.format(o=obj_name))
        return
    return imgpaths


class ObjectMatcher(object):
    def __init__(self, service_name):
        rospy.Service(service_name, ObjectMatch, self._cb_matcher)

    def _cb_matcher(self, req):
        """Callback function for sift match request"""
        rospy.loginfo('Received request: {}'.format(req.objects))
        probs = self.match(req.objects)
        return ObjectMatchResponse(probabilities=probs)

    def match(self, obj_names):
        """Get object match probabilities"""
        raise NotImplementedError('Override this method')


def is_imgfile(filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.jpg', '.jpeg', '.png', '.pgm']:
        return True
    return False


def listdir_for_img(data_dir):
    for f in os.listdir(data_dir):
        if is_imgfile(f):
            yield f

