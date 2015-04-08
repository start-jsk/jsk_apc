#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import gzip
import cPickle as pickle

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import rospy

from matcher_common import get_data_dir, get_object_list


def load_bof_histograms():
    data_dir = get_data_dir()
    bof_path = os.path.join(data_dir, 'bof_data/bof_histograms.pkl.gz')
    with gzip.open(bof_path, 'rb') as f:
        return pickle.load(f)


def main():
    rospy.loginfo('loading bof histograms')
    obj_hists = load_bof_histograms()

    target_names = get_object_list()
    if len(target_names) != len(get_object_list()):
        rospy.logerr('number of objects is invalid')
        return

    X ,y = [], []
    for i, obj_name in enumerate(target_names):
        X.append(obj_hists[obj_name])
        y += [i] * len(obj_hists[obj_name])
    X = np.vstack(X)
    normalize(X, copy=False)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        random_state=np.random.randint(1234))

    lgr = LogisticRegression()
    rospy.loginfo('fitting LogisticRegression')
    lgr.fit(X_train, y_train)
    data_dir = get_data_dir()
    with gzip.open(os.path.join(data_dir, 'bof_data/lgr.pkl.gz'), 'wb') as f:
        pickle.dump(lgr, f)
    y_pred = lgr.predict(X_test)
    rospy.loginfo('score lgr: {}'.format(accuracy_score(y_test, y_pred)))
    rospy.loginfo(classification_report(y_test, y_pred,
                                        target_names=target_names))

    clf = SVC(C=1e4)
    rospy.loginfo('fitting SVM')
    clf.fit(X_train, y_train)
    with gzip.open(os.path.join(data_dir, 'bof_data/svm.pkl.gz'), 'wb') as f:
        pickle.dump(clf, f)
    y_pred = clf.predict(X_test)
    rospy.loginfo('score svm: {}'.format(accuracy_score(y_test, y_pred)))
    rospy.loginfo(classification_report(y_test, y_pred,
                                        target_names=target_names))


if __name__ == '__main__':
    rospy.init_node('train_bof_histogram_clf')
    try:
        main()
    except rospy.ROSInterruptException:
        pass

