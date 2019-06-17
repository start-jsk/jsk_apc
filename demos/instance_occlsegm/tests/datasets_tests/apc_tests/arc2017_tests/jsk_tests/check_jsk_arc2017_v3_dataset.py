if __name__ == '__main__':
    import instance_occlsegm_lib
    dataset = instance_occlsegm_lib.datasets.apc.arc2017.JskARC2017DatasetV3(
        'train', aug='none')
    instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
