if __name__ == '__main__':
    import instance_occlsegm_lib
    dataset = instance_occlsegm_lib.datasets.apc.arc2017.JskARC2017DatasetV1(
        'train')
    instance_occlsegm_lib.datasets.view_class_seg_dataset(dataset)
