from ... import synthetic2d

from .panoptic_occlusion_segmentation import transform_to_panoptic
from .utils import view_panoptic_occlusion_segmentation_dataset


class PanopticOcclusionSegmentationSyntheticDataset(
    synthetic2d.datasets.ARC2017SyntheticInstancesDataset
):

    def __init__(self, size, do_aug=False, aug_level='all'):
        self._size = size
        super(PanopticOcclusionSegmentationSyntheticDataset, self)\
            .__init__(do_aug=do_aug, aug_level=aug_level)

    def __len__(self):
        return self._size

    def get_example(self, i):
        in_data = super(PanopticOcclusionSegmentationSyntheticDataset, self)\
            .get_example(i)
        return transform_to_panoptic(in_data, class_names=self.class_names)


if __name__ == '__main__':
    # import instance_occlsegm_lib
    #
    # dataset = synthetic2d.datasets.ARC2017SyntheticInstancesDataset(
    #     do_aug=False
    # )
    # instance_occlsegm_lib.datasets.view_instance_seg_dataset(
    #     dataset, n_mask_class=3)

    dataset = PanopticOcclusionSegmentationSyntheticDataset(
        size=100, do_aug=False,
    )
    view_panoptic_occlusion_segmentation_dataset(dataset)
