#!/usr/bin/env python

from bvlc_alex import Alex
import chainer
from chainer.links import caffe
from convert_to_chainermodel import convert_to_chainermodel
import os

import rospkg


def convert_alex_caffemodel(caffemodel_path, chainermodel_path):
    chainermodel = Alex()
    caffemodel = caffe.CaffeFunction(caffemodel_path)
    convert_to_chainermodel(caffemodel, chainermodel)
    chainer.serializers.save_hdf5(chainermodel_path, chainermodel)

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    caffemodel_path = os.path.join(
        rospack.get_path('selective_dualarm_stowing'),
        'models/bvlc_alexnet.caffemodel')
    chainermodel_path = os.path.join(
        rospack.get_path('selective_dualarm_stowing'),
        'models/bvlc_alexnet.chainermodel')
    caffemodel_path = os.path.abspath(caffemodel_path)
    if not os.path.exists(caffemodel_path):
        print('Download alexnet caffemodel first')
    if not os.path.exists(chainermodel_path):
        convert_alex_caffemodel(caffemodel_path, chainermodel_path)
