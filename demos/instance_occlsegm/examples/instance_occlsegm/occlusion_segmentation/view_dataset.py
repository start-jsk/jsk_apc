#!/usr/bin/env python

from instance_occlsegm_lib.contrib import instance_occlsegm


if __name__ == '__main__':
    data = instance_occlsegm.datasets.OcclusionSegmentationDataset('train')
    data.split = 'train'
    instance_occlsegm.datasets.view_occlusion_segmentation_dataset(data)
