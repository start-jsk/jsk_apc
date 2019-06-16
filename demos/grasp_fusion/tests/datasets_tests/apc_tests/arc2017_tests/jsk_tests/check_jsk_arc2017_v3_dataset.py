if __name__ == '__main__':
    import grasp_fusion_lib
    dataset = grasp_fusion_lib.datasets.apc.arc2017.JskARC2017DatasetV3(
        'train', aug='none')
    grasp_fusion_lib.datasets.view_class_seg_dataset(dataset)
