#!/usr/bin/env python

import argparse
import json
import os
import os.path as osp
import random
import rospkg

import jsk_arc2017_common


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')


def generate_stow_json(dirname):
    label_list = jsk_arc2017_common.get_object_names()[1:-1]
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
    parser.add_argument('-d', '--dir', default='sample_stow_task')
    args = parser.parse_args()

    dirname = args.dir
    generate_stow_json(dirname)


if __name__ == '__main__':
    main()
