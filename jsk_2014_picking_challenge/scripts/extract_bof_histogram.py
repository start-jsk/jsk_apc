#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
import gzip
import argparse
import cPickle as pickle

from extract_bof import get_sift_descriptors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bof')
    parser.add_argument('siftdata_dir')
    args = parser.parse_args(sys.argv[1:])

    print('loading bof...')
    with gzip.open(args.bof, 'wb') as f:
        bof = pickle.load(f)

    print('making histograms...')
    obj_hists = {}
    obj_descs = get_sift_descriptors(data_dir=args.siftdata_dir)
    for obj, descs in obj_descs:
        obj_hists[obj] = bof.transform(descs)

    print('saving histograms...')
    with gzip.open('bof_histograms.pkl.gz', 'wb') as f:
        pickle.dump(obj_hists, f)


if __name__ == '__main__':
    main()
