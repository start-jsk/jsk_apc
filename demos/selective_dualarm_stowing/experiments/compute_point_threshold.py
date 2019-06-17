#!/usr/bin/env python

import argparse
import chainer
from chainer import cuda
import chainer.serializers as S
from chainer import Variable
from dataset import SinglearmFailureDatasetV4
from models import Alex
from models import VGG16
import numpy as np
import os
import os.path as osp
import re
from utils import get_APC_pt


def compute_point_threshold(
        gpu, data_type, log_dir, model_type):
    answer_pt_list = []
    singlearm_pt_list = []
    dualarm_pt_list = []
    random_state_file = osp.join(log_dir, 'random_state.txt')
    with open(random_state_file) as f:
        random_state = f.read()
    random_state = int(random_state)
    if model_type == 'VGG16':
        model_class = VGG16
        resize_rate = 0.5
    else:
        model_class = Alex
        resize_rate = 1.0
    dataset = SinglearmFailureDatasetV4(
        data_type, random_state=random_state, resize_rate=resize_rate)
    model_name_list = []
    for root, dirs, files in os.walk(log_dir):
        model_name_pattern = '{}_model_iter_'.format(model_type)
        for file_name in files:
            if re.match(model_name_pattern, file_name):
                model_name_list.append(file_name)
    if len(model_name_list) > 1:
        model_iter_list = []
        for file_name in model_name_list:
            model_iter_list.append(
                int(re.split(model_name_pattern, file_name)[1][:-3]))
        model_name = model_name_list[np.argmax(model_iter_list)]
    else:
        model_name = model_name_list[0]
    model_path = osp.join(log_dir, model_name)
    model = model_class(n_class=6)
    S.load_hdf5(model_path, model)
    model.train = True
    model.train_conv = False
    chainer.cuda.get_device(gpu).use()
    model.to_gpu()
    for i in xrange(len(dataset)):
        model.train = False
        x_data, t_data = dataset.get_example(i)
        answer_pt_list.append(get_APC_pt(t_data))
        x_data = np.array([x_data], dtype=np.float32)
        t_data = np.array([t_data], dtype=np.int32)
        x_data = cuda.to_gpu(x_data, device=gpu)
        t_data = cuda.to_gpu(t_data, device=gpu)
        x = Variable(x_data, volatile=True)
        t = Variable(t_data, volatile=True)
        model(x, t)

        h_proba = cuda.to_cpu(model.h_prob.data)
        h_proba = h_proba[0]
        singlearm_proba = h_proba[:3]
        dualarm_proba = h_proba[3:]
        singlearm_pt_list.append(get_APC_pt(singlearm_proba))
        dualarm_pt_list.append(get_APC_pt(dualarm_proba))
    average_answer_pt = sum(answer_pt_list) / len(answer_pt_list)
    average_singlearm_pt = sum(singlearm_pt_list) / len(singlearm_pt_list)
    average_dualarm_pt = sum(dualarm_pt_list) / len(dualarm_pt_list)
    average_answer_pt_diff = average_singlearm_pt - average_answer_pt
    average_pt_diff = average_dualarm_pt - average_singlearm_pt
    print('=========================')
    print('log_dir             : {0}'.format(log_dir))
    print('average answer pt   : {0}'.format(average_answer_pt))
    print('average singlearm pt: {0}'.format(average_singlearm_pt))
    print('average dualarm pt  : {0}'.format(average_dualarm_pt))
    print('single - answer     : {0}'.format(average_answer_pt_diff))
    print('dual - single       : {0}'.format(average_pt_diff))
    print('=========================')
    return average_dualarm_pt, average_singlearm_pt, average_answer_pt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data-type', default='val')
    parser.add_argument('-l', '--log-dir')
    parser.add_argument('--model-type', default='Alex')
    args = parser.parse_args()

    gpu = args.gpu
    data_type = args.data_type
    log_dir = args.log_dir
    model_type = args.model_type

    av_d_pt_list = []
    av_s_pt_list = []
    av_ans_pt_list = []
    for root, dirs, files in os.walk(log_dir):
        for i in xrange(len(dirs)):
            tmp_log_dir = osp.join(log_dir, '{0:02d}'.format(i))
            av_d_pt, av_s_pt, av_ans_pt = compute_point_threshold(
                gpu, data_type, tmp_log_dir, model_type)
            av_d_pt_list.append(av_d_pt)
            av_s_pt_list.append(av_s_pt)
            av_ans_pt_list.append(av_ans_pt)
    final_av_d_pt = sum(av_d_pt_list) / len(av_d_pt_list)
    final_av_s_pt = sum(av_s_pt_list) / len(av_s_pt_list)
    final_av_ans_pt = sum(av_ans_pt_list) / len(av_ans_pt_list)
    final_ans_pt_diff = final_av_s_pt - final_av_ans_pt
    final_pt_diff = final_av_d_pt - final_av_s_pt
    print('=========================')
    print('final result')
    print('average answer pt   : {0}'.format(final_av_ans_pt))
    print('average singlearm pt: {0}'.format(final_av_s_pt))
    print('average dualarm pt  : {0}'.format(final_av_d_pt))
    print('single - ans        : {0}'.format(final_ans_pt_diff))
    print('dual - single       : {0}'.format(final_pt_diff))
    print('=========================')
