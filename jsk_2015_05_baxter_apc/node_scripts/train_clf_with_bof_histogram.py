#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import sys
import gzip
import cPickle as pickle
import argparse

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from common import get_object_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bof_histogram')
    args = parser.parse_args(sys.argv[1:])

    print('loading bof histogram')
    with gzip.open(args.bof_histogram, 'rb') as f:
        obj_hists = pickle.load(f)

    target_names = get_object_list()

    # create train and test data
    X ,y = [], []
    for i, obj_name in enumerate(target_names):
        X.append(obj_hists[obj_name])
        y += [i] * len(obj_hists[obj_name])
    X = np.vstack(X)
    normalize(X, copy=False)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        random_state=np.random.randint(1234))

    # train and test
    lgr = LogisticRegression()
    print('fitting LogisticRegression')
    lgr.fit(X_train, y_train)
    with gzip.open('lgr.pkl.gz', 'wb') as f:
        pickle.dump(lgr, f)
    y_pred = lgr.predict(X_test)
    print('score lgr: {}'.format(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred,
                                        target_names=target_names))


if __name__ == '__main__':
    main()
