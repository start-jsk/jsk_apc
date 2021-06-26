#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle as pickle
import gzip
import sys

import cv2
from imagesift import get_sift_keypoints
import numpy as np
from sklearn.datasets import load_files
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize


parser = argparse.ArgumentParser()
parser.add_argument('container_path')
parser.add_argument('bof_path')
parser.add_argument('clf_path')
args = parser.parse_args()

container_path = args.container_path
bof_path = args.bof_path
clf_path = args.clf_path

bunch_files = load_files(container_path=container_path,
                         description='images',
                         shuffle=False,
                         load_content=False)

with gzip.open(bof_path, 'rb') as f:
    bof = pickle.load(f)

with gzip.open(clf_path, 'rb') as f:
    clf = pickle.load(f)

descs = []
for fname in bunch_files.filenames:
    img = cv2.imread(fname, 0)
    _, desc = get_sift_keypoints(img)
    descs.append(desc)
X = bof.transform(descs)
normalize(X, copy=False)
y_pred = clf.predict(X)

y = bunch_files.target
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred, target_names=clf.target_names_))
