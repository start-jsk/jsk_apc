if __name__ == '__main__':
    import instance_occlsegm_lib
    dataset = instance_occlsegm_lib.datasets.coco.COCOClassSegmentationDataset(
        'minival')
    instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
