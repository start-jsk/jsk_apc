#!/usr/bin/env python

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2016)
    args = parser.parse_args()

    if args.year == 2015:
        import jsk_apc2015_common
        cls_names = ['background'] + jsk_apc2015_common.get_object_list()
    elif args.year == 2016:
        import jsk_apc2016_common
        data = jsk_apc2016_common.get_object_data()
        cls_names = ['background'] + [d['name'] for d in data]
    else:
        raise ValueError

    text = []
    for cls_id, cls_name in enumerate(cls_names):
        text.append('{:2}: {}'.format(cls_id, cls_name))
    print('\n'.join(text))


if __name__ == '__main__':
    main()
