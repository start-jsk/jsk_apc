#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import sys
import gzip
import argparse
import cPickle as pickle

import numpy as np

from common import get_object_list, load_siftdata
from bag_of_features import BagOfFeatures


def get_sift_descriptors(n_imgs=None, data_dir=None):
    objects = get_object_list()
    obj_descs = []
    for obj in objects:
        descs = load_siftdata(obj_name=obj,
                              return_pos=False,
                              data_dir=data_dir)
        if descs is None:
            continue
        if n_imgs is None:
            n_imgs = len(descs)
        p = np.random.randint(0, len(descs), size=n_imgs)
        descs = np.array(map(lambda x: x.astype('float16'), descs))
        obj_descs.append((obj, descs[p]))
    return obj_descs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('siftdata_dir')
    parser.add_argument('n_imgs')
    args = parser.parse_args(sys.argv[1:])

    print('getting descriptors...')
    sift_descs = get_sift_descriptors(n_imgs=int(args.n_imgs),
                                      data_dir=args.siftdata_dir)
    _, descs = zip(*sift_descs)
    X = []
    for d in descs:
        xi = np.vstack(map(lambda x: x.reshape((-1, 128)), d))
        X.append(xi)
    X = np.vstack(X)
    np.random.shuffle(X)
    print('X.shape: {}'.format(X.shape))

    print('fitting bag of features...')
    bof = BagOfFeatures()
    bof.fit(X)
    filename = 'bof_{0}.pkl.gz'.format(
        hashlib.sha1(str(time.time())).hexdigest()[:8])
    with gzip.open('bof.pkl.gz', 'wb') as f:
        pickle.dump(bof, f)


if __name__ == '__main__':
    main()
