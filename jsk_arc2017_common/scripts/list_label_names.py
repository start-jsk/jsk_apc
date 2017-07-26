#!/usr/bin/env python

import re

from termcolor import cprint

import jsk_arc2017_common


def main():
    known_object_names = jsk_arc2017_common.get_known_object_names()
    n_unknown = 0
    for obj_id, obj in enumerate(jsk_arc2017_common.get_label_names()):
        msg = '{:02}: {}'.format(obj_id, obj)
        if obj in known_object_names or re.match('__(.*?)__', obj):
            print(msg)
        else:
            cprint(msg, color='red')
            n_unknown += 1
    cprint('Number of unknown objects: %d' % n_unknown, color='red')


if __name__ == '__main__':
    main()
