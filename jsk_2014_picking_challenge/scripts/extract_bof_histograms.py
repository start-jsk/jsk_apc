#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function
import os
import gzip
import cPickle as pickle

import numpy as np

import rospy

from bag_of_features import BagOfFeatures
from matcher_common import load_siftdata, get_object_list, get_data_dir


def get_descriptors(n_imgs=None, cache=True):
    data_dir = get_data_dir()
    cache_file = os.path.join(data_dir,
        'bof_data/obj_descs_{}.pkl.gz'.format(n_imgs))
    if cache and os.path.exists(cache_file):
        # load cache
        with gzip.open(cache_file, 'rb') as f:
            return pickle.load(f)
    # without cache
    objects = get_object_list()
    obj_descs = []
    for obj in objects:
        descs = load_siftdata(obj_name=obj, return_pos=False)
        if descs is None:
            continue
        if n_imgs is None:
            n_imgs = len(descs)
        p = np.random.randint(0, len(descs), size=n_imgs)
        descs = np.array(map(lambda x: x.astype('float16'), descs))
        obj_descs.append((obj, descs[p]))
    # store cache
    with gzip.open(cache_file, 'wb') as f:
        pickle.dump(obj_descs, f)
    return obj_descs


def extract_bof_histograms():
    rospy.loginfo('getting descriptors...')
    _, descs = zip(*get_descriptors(n_imgs=30))
    X = []
    for d in descs:
        X.append(np.vstack(map(lambda x: x.reshape((-1, 128)), d)))
    del descs
    X = np.vstack(X)
    np.random.shuffle(X)
    rospy.loginfo('X.shape: {}'.format(X.shape))

    rospy.loginfo('fitting BagOfFeatures...')
    bof = BagOfFeatures()
    bof.fit(X)
    del X

    rospy.loginfo('making histograms...')
    obj_hists = {}
    obj_descs = get_descriptors(n_imgs=100)
    for obj, descs in obj_descs:
        obj_hists[obj] = bof.transform(descs)
    del descs

    def save_bof_histograms(hists):
        data_dir = get_data_dir()
        bof_hist_path = os.path.join(data_dir,
                                     'bof_data/bof_histograms.pkl.gz')
        with gzip.open(bof_hist_path, 'wb') as f:
            pickle.dump(hists, f)

    rospy.loginfo('saving histograms...')
    save_bof_histograms(obj_hists)

    data_dir = get_data_dir()
    bof.save_bof(path=os.path.join(data_dir, 'bof_data/bof.pkl.gz'))


if __name__ == '__main__':
    rospy.init_node('extract_bof_histograms')
    try:
        extract_bof_histograms()
    except rospy.ROSInterruptException:
        pass

