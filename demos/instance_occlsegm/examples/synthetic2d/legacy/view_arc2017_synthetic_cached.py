#!/usr/bin/env python

import instance_occlsegm_lib

from contrib.datasets import ARC2017SyntheticCachedDataset


dataset = ARC2017SyntheticCachedDataset(split='test')
instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
