#!/usr/bin/env python

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib import synthetic2d


if __name__ == '__main__':
    dataset = synthetic2d.datasets.InstanceImageDataset(load_pred=False)
    instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
