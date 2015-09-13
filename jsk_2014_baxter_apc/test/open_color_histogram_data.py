#!/usr/bin/env python
# -*- coding:utf-8 -*-

import gzip
import cPickle
import sys
import argparse

def main():
    arg_fmt = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-f', '--filename', required=True,
        help='select file name.'
    )
    args = parser.parse_args()
    with gzip.open('../data/histogram_data/' + args.filename + '.pkl.gz', 'rb') as gf:
        data = cPickle.load(gf)
    print(data)
    print(len(data[0]))

if __name__ == "__main__":
    main()
