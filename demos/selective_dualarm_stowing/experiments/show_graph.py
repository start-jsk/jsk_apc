#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from selective_arm_stowing.dataset.v4 import DualarmDatasetV4


# Python2's xrange equals Python3's range, and xrange is removed on Python3
if not hasattr(__builtins__, 'xrange'):
    xrange = range


def show_acc_graph(log_dir, line_num):
    for root, dirs, files in os.walk(log_dir):
        plt.figure()
        # plt.style.use('seaborn-whitegrid')
        column_num = len(dirs) / line_num
        if len(dirs) % line_num != 0:
            column_num += 1
        for i in xrange(len(dirs)):
            log_file = os.path.join(root, '{0:02d}'.format(i), 'log.json')
            df = pd.read_json(log_file)
            train_acc = df['main/acc']
            train_loss = df['main/loss']
            try:
                val_acc = df['validation/main/acc']
                val_acc = val_acc[~pd.isnull(val_acc)]
            except Exception:
                val_acc = None

            # plot acc graph

            host = plt.subplot(column_num, line_num, i+1)
            par_loss = host.twinx()

            # plt.title('loop:{0:02d}'.format(i))
            host.plot(train_acc, 'b-', lw=0.5, label='Training Accuracy')
            host.set_xlim(0, len(train_acc))
            host.set_yticks(np.arange(0, 1, 0.2))
            host.set_ylim(0, 1)
            host.grid(which='major', linestyle=':')
            if val_acc is not None:
                host.plot(
                    val_acc, 'bo', markersize=3, label='Validation Accuracy')
            par_loss.plot(train_loss, 'r-', lw=0.5, label='Training Loss')
            par_loss.set_xlim(0, len(train_loss))
            par_loss.set_yticks(np.arange(0, 1.0, 0.2))
            par_loss.set_ylim(0, 1.0)
            par_loss.grid(which='major', linestyle=':')

            host.set_xlabel('Iteration', fontsize=6)
            host.set_ylabel('Accuracy', fontsize=6)
            par_loss.set_ylabel('Loss', fontsize=6)
            host.legend(loc="center right", fontsize=5)
            par_loss.legend(
                loc="center right", bbox_to_anchor=(1.0, 0.3), fontsize=5)
            host.tick_params(labelsize=6)
            par_loss.tick_params(labelsize=6)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.6, hspace=0.3)
        plt.show()


def get_val_acc(log_dir):
    val_acc_list = []
    for root, dirs, files in os.walk(log_dir):
        for i in xrange(len(dirs)):
            log_file = os.path.join(root, '{0:02d}'.format(i), 'log.json')
            df = pd.read_json(log_file)
            try:
                val_acc = df['validation/main/acc']
                val_acc = val_acc[~val_acc.isnull()].as_matrix()
                if len(val_acc) > 0:
                    val_acc_list.append(val_acc.tolist())
            except Exception:
                continue
    return val_acc_list


def get_baseline_acc(log_dir):
    baseline_acc_list = []
    for root, dirs, files in os.walk(log_dir):
        for i in xrange(len(dirs)):
            random_state_file = os.path.join(
                root, '{0:02d}'.format(i), 'random_state.txt')
            try:
                with open(random_state_file) as f:
                    random_state = f.read()
                dataset_val = DualarmDatasetV4(
                    'val', int(random_state), 1, 0.3)
                baseline_acc = dataset_val.get_baseline_acc()
                baseline_acc_list.append(baseline_acc)
            except Exception:
                continue
    return baseline_acc_list


def compare_acc(log_dir):
    val_acc_list = get_val_acc(log_dir)
    final_val_acc_list = [x[-1] for x in val_acc_list]
    baseline_acc_list = get_baseline_acc(log_dir)
    average_final_val_acc = sum(final_val_acc_list) / len(final_val_acc_list)
    average_baseline_acc = sum(baseline_acc_list) / len(baseline_acc_list)
    # for final_val_acc, baseline_acc in zip(
    #         final_val_acc_list, baseline_acc_list):
    #     print('{0:.2f} > {1:.2f}'.format(final_val_acc, baseline_acc))
    inversed_val_acc_list = []
    for i in xrange(len(val_acc_list[0])):
        tmp_list = []
        for val_acc in val_acc_list:
            try:
                tmp_list.append(val_acc[i])
            except Exception:
                continue
        inversed_val_acc_list.append(tmp_list)
    average_val_acc_list = [sum(x) / len(x) for x in inversed_val_acc_list]
    max_index = np.array(average_val_acc_list).argmax()
    if max_index != len(average_val_acc_list) - 1:
        print('WARNING: val_acc[{}] average > final_val_acc average'.format(
            max_index))
        print('val_acc[{0}] average   : {1}'.format(
            max_index, max(average_val_acc_list)))
    print('final_val_acc average: {}'.format(average_final_val_acc))
    print('baseline_acc average : {}'.format(average_baseline_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', '-l', default='./out/alex/latest')
    args = parser.parse_args()
    log_dir = args.log_dir
    compare_acc(log_dir)
    show_acc_graph(log_dir, 5)
