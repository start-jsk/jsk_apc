#!/usr/bin/env python

import argparse
import json
import os.path as osp
import random
import rospkg
import yaml


PKG_DIR = rospkg.RosPack().get_path('jsk_arc2017_common')


def load_label_list():
    with open(osp.join(PKG_DIR, 'config', 'label_names.yaml')) as f:
        label_names = yaml.load(f)
    label_list = label_names['label_names']
    label_list = label_list[1:-1]
    return label_list


def main(filename):
    label_list = load_label_list()
    bin_contents = random.sample(label_list, 32)
    target_items = random.sample(bin_contents, 10)
    random.shuffle(bin_contents)
    bin_A_contents = bin_contents[:16]
    bin_B_contents = bin_contents[16:24]
    bin_C_contents = bin_contents[24:32]
    random.shuffle(target_items)
    box_A_contents = target_items[:5]
    box_B_contents = target_items[5:8]
    box_C_contents = target_items[8:10]
    output = {
        "bin_contents": {
            "bin_A": bin_A_contents,
            "bin_B": bin_B_contents,
            "bin_C": bin_C_contents
        },
        "target_items": {
            "box_A": box_A_contents,
            "box_B": box_B_contents,
            "box_C": box_C_contents
        },

    }
    pick_json_path = osp.join(PKG_DIR, 'json', filename)
    with open(pick_json_path, 'w+') as f:
        json.dump(output, f, sort_keys=True, indent=4, separators=(',', ': '))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='sample_pick.json')
    args = parser.parse_args()

    filename = args.output
    main(filename)
