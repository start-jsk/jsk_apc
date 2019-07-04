#!/usr/bin/env python

import argparse
import os
import os.path as osp


def analyze_dataset(dataset_dir):
    dataset_analysis = {
        'singlearm': {
            'total_num': 0,
            'success': 0,
            'drop': 0,
            'protrude': 0,
            'damage': 0},
        'dualarm': {
            'total_num': 0,
            'success': 0,
            'drop': 0,
            'protrude': 0,
            'damage': 0}
        }
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            dir_path = osp.join(root, dir_name)
            # with open(osp.join(dir_path, 'target.txt')) as target_f:
            #     target_name = target_f.read()
            with open(osp.join(dir_path, 'label.txt')) as label_f:
                label_list = label_f.read()
            label_list = [x for x in label_list.split('\n') if len(x) > 0]
            motion_name = label_list[0].split('_')[0]
            result_list = [x.split('_')[1] for x in label_list]
            dataset_analysis[motion_name]['total_num'] += 1
            for result in result_list:
                dataset_analysis[motion_name][result] += 1
    for motion_name in ['singlearm', 'dualarm']:
        motion_analysis = dataset_analysis[motion_name]
        print('===================')
        print('Motion: {}'.format(motion_name))
        print('Total    : {}'.format(motion_analysis['total_num']))
        print('Success  : {}'.format(motion_analysis['success']))
        print('Drop     : {}'.format(motion_analysis['drop']))
        print('Protrude : {}'.format(motion_analysis['protrude']))
        print('Damage   : {}'.format(motion_analysis['damage']))
        print('===================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d')
    args = parser.parse_args()
    analyze_dataset(args.dir)
