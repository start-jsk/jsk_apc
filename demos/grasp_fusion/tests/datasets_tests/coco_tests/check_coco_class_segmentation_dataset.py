if __name__ == '__main__':
    import grasp_fusion_lib
    dataset = grasp_fusion_lib.datasets.coco.COCOClassSegmentationDataset(
        'minival')
    grasp_fusion_lib.datasets.view_class_seg_dataset(dataset)
