#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
import argparse
import gzip
import cPickle as pickle
import numpy as np
import time
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('histogram1')
parser.add_argument('histogram2')
args = parser.parse_args(sys.argv[1:])

with gzip.open(args.histogram1) as f:
    histogram1 = pickle.load(f)
with gzip.open(args.histogram2) as f:
    histogram2 = pickle.load(f)

hist_merged = {}
for obj in histogram1:
    if obj in histogram1 and obj in histogram2:
        hist = np.vstack([histogram1[obj], histogram2[obj]])
    elif obj in histogram1:
        hist = histogram1[obj]
    elif obj in histogram2:
        hist = histogram2[obj]
    else:
        hist = np.array([])
    hist_merged[obj] = hist

filename = 'histogram_merged_{0}.pkl.gz'.format(
    hashlib.sha1(str(time.time())).hexdigest()[:8])
with gzip.open(filename, 'wb') as f:
    pickle.dump(hist_merged, f)
