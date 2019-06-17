#!/usr/bin/env python

import instance_occlsegm_lib

from contrib.datasets import ARC2017RealDataset


dataset = ARC2017RealDataset(split='train')
instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
