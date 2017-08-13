#!/usr/bin/env python

import json
import os
import os.path as osp


here = osp.dirname(osp.abspath(__file__))


def main():
    item_data_dir = osp.join(here, 'item_data')

    for item_name_upper in os.listdir(item_data_dir):
        item_dir = osp.join(item_data_dir, item_name_upper)
        if not osp.isdir(item_dir):
            continue

        item_name_lower = item_name_upper.lower()
        print('{:s}: {:s}'.format(item_name_lower, item_dir))

        json_file = osp.join(item_dir, '{:s}.json'.format(item_name_lower))
        data_origin = {}
        if osp.exists(json_file):
            data_origin = json.load(open(json_file))
        with open(json_file, 'w') as f:
            data = {
                'name': item_name_lower,
                "weight": None,
                "dimensions": [None, None, None],
                "type": None,
                "description": None,
            }
            data.update(data_origin)
            json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
