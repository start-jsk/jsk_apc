#!/usr/bin/env python

from __future__ import print_function

import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    fname = parser.parse_args().filename

    f = open(fname, 'r')
    line = f.readline()
    cnt = 0
    while line:
        if cnt == 0:
            if 'range_millimeter' in line:
                print('{},'.format(
                    yaml.load(line,
                              Loader=yaml.SafeLoader)['range_millimeter']),
                      end='')
                cnt = cnt + 1
        elif cnt == 1:
            if 'distance' in line:
                print('{},'.format(
                    yaml.load(line, Loader=yaml.SafeLoader)['distance']),
                      end='')
                cnt = cnt + 1
        elif cnt == 2:
            if 'diff_from_init' in line:
                diff = yaml.load(line,
                                 Loader=yaml.SafeLoader)['diff_from_init']
            # elif 'init_value' in line:
            #     init = yaml.load(line, Loader=yaml.SafeLoader)['init_value']
            #     print(init + diff)
                print(diff)
                cnt = 0
        line = f.readline()
    f.close()


if __name__ == '__main__':
    main()