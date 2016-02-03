#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
this script will extract color_histogram from masked_data($(jsk_2015_05_baxter_apc)/data/masked_data)
and save it.
"""

import numpy as np
import os
import gzip
import cPickle as pickle

from sklearn.svm import SVC
from skimage.io import imread

from common import get_data_dir

import jsk_apc2015_common


def get_imlist(path, extension='jpg'):
    """ return all image files list in path"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.' + extension)]

class ColorHistogramFeatures(object):
    def __init__(self):
        self.file_name = 'rgb'
        self.object_names = jsk_apc2015_common.get_object_list()
        self.cfeatures = []
        self.labels = []
    def save_data(self):
        print('saving data')
        with gzip.open(self.file_name + '.pkl.gz', 'wb') as f:
            pickle.dump(self.cfeatures, f)
            pickle.dump(self.labels, f)
        print("saved data")
    def load_data(self):
        '''  from pkl file load feature and label data '''
        print("loading data ... ")
        data_dir = get_data_dir()
        feature_path = os.path.join(data_dir, 'histogram_data/rgb.pkl.gz')
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
        im = imread(im)
        return self.color_hist(im)
    def color_hist(self, im):
        ''' Compute color histogram of input image '''
        # Downsample pixel values:
        im = im // 64

        # Separate RGB channels:
        r,g,b = im.transpose((2,0,1))

        pixels = 1 * r + 4 * g + 16 * b
        hist = np.bincount(pixels.ravel(), minlength=64)
        hist = hist.astype(float)
        return np.log1p(hist)
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
