#!/usr/bin/env python

import argparse
import json
import os
import os.path as osp
import random
import rospkg
import shutil

import jsk_arc2017_common


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')


def generate_pick_json(dirname=None):
    if dirname is None:
        dirname = 'sample_pick_task'
    label_list = jsk_arc2017_common.get_object_names()
    box_id_list = ['A1', '1AD', '1A5', '1B2', 'K3']
    box_A_candidate = box_id_list[:2]   # box_A is for 2 items
    box_B_candidate = box_id_list[2:4]  # box_B is for 3 items
    box_C_candidate = box_id_list[3:5]  # box_C is for 5 items
    box_A_id = random.sample(box_A_candidate, 1)[0]
    box_B_id = random.sample(box_B_candidate, 1)[0]
    if box_B_id == '1B2':
        box_C_id = 'K3'
    else:
        box_C_id = random.sample(box_C_candidate, 1)[0]

    bin_contents = random.sample(label_list, 32)
    target_items = random.sample(bin_contents, 10)
    random.shuffle(bin_contents)
    bin_A_contents = bin_contents[:10]
    bin_B_contents = bin_contents[10:22]
    bin_C_contents = bin_contents[22:32]
    random.shuffle(target_items)
    box_A_contents = target_items[:2]
    box_B_contents = target_items[2:5]
    box_C_contents = target_items[5:10]
    order = {
        'orders': [
            {
                'size_id': box_A_id,
                'contents': box_A_contents
            },
            {
                'size_id': box_B_id,
                'contents': box_B_contents
            },
            {
                'size_id': box_C_id,
                'contents': box_C_contents
            },
        ]
    }
    location = {
        'bins': [
            {
                'bin_id': 'A',
                'contents': bin_A_contents
            },
            {
                'bin_id': 'B',
                'contents': bin_B_contents
            },
            {
                'bin_id': 'C',
                'contents': bin_C_contents
            }
        ],
        'boxes': [
            {
                'size_id': box_A_id,
                'contents': [
                ]
            },
            {
                'size_id': box_B_id,
                'contents': [
                ]
            },
            {
                'size_id': box_C_id,
                'contents': [
                ]
            }
        ],
        'tote': {
            'contents': [
            ]
        }
    }

    separators = (',', ': ')
    original_box_sizes_path = osp.join(PKG_DIR, 'config', 'box_sizes.json')
    output_dir = osp.join(PKG_DIR, 'data', 'json', dirname)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    else:
        raise IOError('output dir already exists: {}'.format(output_dir))

    box_sizes_path = osp.join(output_dir, 'box_sizes.json')
    shutil.copyfile(original_box_sizes_path, box_sizes_path)

    location_path = osp.join(output_dir, 'item_location_file.json')
    with open(location_path, 'w+') as f:
        json.dump(location, f, sort_keys=True, indent=4, separators=separators)

    order_path = osp.join(output_dir, 'order_file.json')
    with open(order_path, 'w+') as f:
        json.dump(order, f, sort_keys=True, indent=4, separators=separators)


def generate_stow_json(dirname):
    if dirname is None:
        dirname = 'sample_stow_task'
    label_list = jsk_arc2017_common.get_object_names()
    tote_contents = random.sample(label_list, 20)
    location = {
        'bins': [
            {
                'bin_id': 'A',
                'contents': []
            },
            {
                'bin_id': 'B',
                'contents': []
            },
            {
                'bin_id': 'C',
                'contents': []
            }
        ],
        'boxes': [],
        'tote': {
            'contents': tote_contents
        }
    }

    separators = (',', ': ')
    output_dir = osp.join(PKG_DIR, 'data', 'json', dirname)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    else:
        raise IOError('output dir already exists: {}'.format(output_dir))

    location_path = osp.join(output_dir, 'item_location_file.json')
    with open(location_path, 'w+') as f:
        json.dump(location, f, sort_keys=True, indent=4, separators=separators)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default=None, help='output JSON dir')
    parser.add_argument(
        '-s', '--stow', action='store_true', help='generate stow task JSON')
    args = parser.parse_args()

    dirname = args.dir
    if args.stow:
        generate_stow_json(dirname)
    else:
        generate_pick_json(dirname)


if __name__ == '__main__':
    main()
