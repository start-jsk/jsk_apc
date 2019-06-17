#!/usr/bin/env python

import instance_occlsegm_lib


def main():
    dataset = instance_occlsegm_lib.datasets.apc.\
        ARC2017InstanceSegmentationDataset(split='train', aug='standard')

    instance_occlsegm_lib.datasets.view_instance_seg_dataset(dataset)


if __name__ == '__main__':
    main()
