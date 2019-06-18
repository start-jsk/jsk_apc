# Disable this test because it's too slow
#
# import numpy as np
#
# import grasp_fusion_lib
#
#
# def test_voc_class_segmentation_dataset(split):
#     for split in ['train', 'val']:
#         dataset = grasp_fusion_lib.datasets.voc.VOCClassSegmentationDataset(
#             split=split)
#         assert len(dataset) > 0
#
#         assert hasattr(dataset, 'class_names')
#         assert hasattr(dataset, '__getitem__')
#         assert hasattr(dataset, '__len__')
#
#         img, lbl = dataset[0]
#         assert img.shape[:2] == lbl.shape
#         assert img.shape[2] == 3
#         assert img.dtype == np.uint8
#         assert lbl.dtype == np.int32
