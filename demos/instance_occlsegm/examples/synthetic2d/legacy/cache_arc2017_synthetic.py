#!/usr/bin/env python

import os
import os.path as osp

import concurrent.futures
import numpy as np
import tqdm

import contrib


def cache_in_data(args):
    dataset, index, cache_dir = args

    img, lbl = dataset[index]

    cache_file = osp.join(cache_dir, '%08d.npz' % index)
    np.savez_compressed(cache_file, img=img, lbl=lbl)


# https://techoverflow.net/2017/05/18/how-to-use-concurrent-futures-map-with-a-tqdm-progress-bar/  # NOQA
def tqdm_parallel_map(executor, fn, *iterables, **kwargs):
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]
    for f in tqdm.tqdm(concurrent.futures.as_completed(futures_list),
                       total=len(futures_list), **kwargs):
        yield f.result()


if __name__ == '__main__':
    cache_dir = osp.expanduser('~/data/instance_occlsegm_lib/synthetic2d/ARC2017SyntheticCachedDataset')  # NOQA
    try:
        os.makedirs(cache_dir)
    except OSError:
        pass
    print('Caching ARC2017SyntheticDataset to: %s' % cache_dir)

    dataset = contrib.datasets.ARC2017SyntheticDataset(
        do_aug=True, aug_level='object')

    executor = concurrent.futures.ProcessPoolExecutor()
    iterables = ((dataset, i, cache_dir) for i in range(len(dataset)))
    results = tqdm_parallel_map(executor, cache_in_data, iterables)

    # wait for results
    for res in results:
        pass

    print('==> Done!')
