#!/usr/bin/env python

import instance_occlsegm_lib

from contrib.datasets import ARC2017SyntheticDataset


dataset = ARC2017SyntheticDataset()
instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
