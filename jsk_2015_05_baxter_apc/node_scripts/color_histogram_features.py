#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
this script will extract color_histogram from masked_data($(jsk_2015_05_baxter_apc)/data/masked_data)
and save it.
"""
from __future__ import division
import numpy as np
import os
import gzip
import cPickle as pickle
import cv2
import numpy.ma as ma
from sklearn.svm import SVC
from skimage.io import imread

from common import get_data_dir

import jsk_apc2015_common


def get_imlist(path, extension='jpg'):
    """ return all image files list in path"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.' + extension)]

class ColorHistogramFeatures(object):
    def __init__(self):
        self.file_name = 'hsv'
        self.object_names = jsk_apc2015_common.get_object_list()
        self.cfeatures = []
        self.labels = []
    def save_data(self):
        print('saving data')
        data_dir = get_data_dir()
        with gzip.open(os.path.join(data_dir,'histogram_data/',self.file_name) + '.pkl.gz', 'wb') as f:
            pickle.dump(self.cfeatures, f)
            pickle.dump(self.labels, f)
        print("saved data")
    def load_data(self):
        '''  from pkl file load feature and label data '''
        print("loading data ... ")
        data_dir = get_data_dir()
        feature_path = os.path.join(data_dir, 'histogram_data/hsv.pkl.gz')
        with gzip.open(feature_path) as f:
            self.cfeatures = pickle.load(f)
            self.labels = pickle.load(f)
        print("load end.")
        self.init_estimate()
    def images(self, dir_name='masked_data'):
        for object_name in self.object_names:
            print("processing ... {}".format(object_name))
            images_path = get_imlist(os.path.join(os.path.dirname(__file__), '../data', dir_name, object_name))
            for image_path in images_path:
                yield image_path, object_name
    def features_for(self, im):
        im = cv2.imread(im)##previous : im = imread(im). This might cause generating wrong histogram, because skiimage's rgb color space seems different from opencv's rgb color space.
        return self.color_hist(im)
    def color_hist(self, im):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        '''
        0<h<180 , 0<s<255, 0<v<255
        v < 20*2.55=51 : pure black
        else s < 10*25.5=25 :white - gray
        '''
        #Separate HSV channels:
        h,s,v = im.transpose((2,0,1))
        h = h // 4
        mask_v=v[:,:] < 51
        mask_s=s[:,:] < 25
        total_mask=mask_v | mask_s
        masked_h=ma.masked_array(h,total_mask)
        hist_h = np.bincount(masked_h.compressed(), minlength=64)
        hist = hist_h
        hist = hist.astype(float)
        #hist_sum = sum(hist)
        #hist = hist / hist_sum
        return np.sqrt(hist)

    def compute_texture(self):
        print("Computing whole-image color features ... ")
        for im, ell in self.images():
            self.cfeatures.append(self.features_for(im))
            self.labels.append(ell)
        self.cfeatures = np.array(self.cfeatures)
        self.labels = np.array(self.labels)
        # save features data
        self.save_data()
    def init_estimate(self):
        self.estimator = SVC(kernel='linear', probability=True)
        self.estimator.fit(self.cfeatures, self.labels)
    def predict(self, imgs, obj_names=None):
        if obj_names is None:
            obj_names = self.object_names
        im_feature = np.array([self.color_hist(img) for img in imgs])
        return self.estimator.predict_proba(im_feature)
    def run(self):
        self.compute_texture()

def main():
    c = ColorHistogramFeatures()
    c.run()

if __name__ == "__main__":
    main()
