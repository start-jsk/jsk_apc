#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import gzip
import cPickle as pickle

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans


class BagOfFeatures(object):
    def __init__(self, hist_size=500, bof_path=None):
        self.nn = None
        if bof_path is not None:
            self.load_bof(bof_path)
        self.hist_size = hist_size

    def fit(self, X):
        """Fit features and extract bag of features"""
        k = self.hist_size
        km = MiniBatchKMeans(n_clusters=k, init_size=3*k, max_iter=300)
        km.fit(X)
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(km.cluster_centers_)
        self.nn = nn

    def transform(self, X):
        return np.array([self.make_hist(xi.reshape((-1, 128))) for xi in X])

    def make_hist(self, descriptors):
        """Make histogram for one image"""
        nn = self.nn
        if nn is None:
            raise ValueError('must fit features before making histogram')
        indices = nn.kneighbors(descriptors, return_distance=False)
        histogram = np.zeros(self.hist_size)
        for idx in np.unique(indices):
            mask = indices == idx
            histogram[idx] = mask.sum()  # count the idx
            indices = indices[mask == False]
        return histogram

    def save_bof(self, path='bof.pkl.gz'):
        nn = self.nn
        if nn is None:
            raise ValueError('must fit features before saving bof')
        with gzip.open(path, 'wb') as f:
            pickle.dump(nn._fit_X, f)

    def load_bof(self, path='bof.pkl.gz'):
        with gzip.open(path, 'rb') as f:
            bof = pickle.load(f)
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(bof)
        self.nn = nn

