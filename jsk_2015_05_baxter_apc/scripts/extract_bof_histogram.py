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
    parser.add_argument('bof_extractor')
    parser.add_argument('siftdata_dir')
    args = parser.parse_args(sys.argv[1:])

    print('loading bof_extractor...')
    with gzip.open(args.bof_extractor, 'rb') as f:
        bof = pickle.load(f)

    print('making histograms...')
    obj_hists = {}
    obj_descs = get_sift_descriptors(data_dir=args.siftdata_dir)
    for obj, descs in obj_descs:
        obj_hists[obj] = bof.transform(descs)

    print('saving histograms...')
    filename = 'bof_histograms_{0}.pkl.gz'.format(
        hashlib.sha1(str(time.time())).hexdigest()[:8])
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj_hists, f)


if __name__ == '__main__':
    main()
