import numpy as np

from .... import image as image_module
from ..arc2017.jsk import JskARC2017DatasetV3


class ARC2017SemanticSegmentationDataset(JskARC2017DatasetV3):

    def __init__(self, split):
        assert split in ('train', 'test')
        if split == 'test':
            split = 'valid'
        super(ARC2017SemanticSegmentationDataset, self).__init__(
            split, aug='none'
        )
        self.class_names = self.class_names[:-1]  # drop 41: __shelf__

    def get_example(self, i):
        img, lbl = super(ARC2017SemanticSegmentationDataset, self).get_example(
            i
        )
        lbl[lbl == 41] = 0
        return img, lbl


class ARC2017InstanceSegmentationDataset(JskARC2017DatasetV3):

    def __init__(self, split, aug='none'):
        assert aug in ['none', 'standard']
        assert split in ('train', 'test')
        if split == 'test':
            split = 'valid'
        super(ARC2017InstanceSegmentationDataset, self).__init__(
            split, aug=aug
        )
        # drop 0,41: __background__, __shelf__
        self.class_names = self.class_names[1:-1]

    def get_example(self, i):
        img, lbl = super(ARC2017InstanceSegmentationDataset, self).get_example(
            i
        )

        bin_mask = lbl == 41
        y1, x1, y2, x2 = image_module.masks_to_bboxes([bin_mask])[0]
        img = img[y1:y2, x1:x2]
        lbl = lbl[y1:y2, x1:x2]

        img[lbl == -1] = 0
        lbl[lbl == 41] = 0

        class_ids = np.unique(lbl)
        labels, bboxes, masks = [], [], []
        for class_id in class_ids:
            if class_id in [-1, 0]:
                continue

            labels.append(class_id)

            mask = lbl == class_id
            masks.append(mask)
        bboxes = image_module.masks_to_bboxes(masks)

        bboxes = np.asarray(bboxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32) - 1  # skip __background__
        masks = np.asarray(masks, dtype=np.int32)

        return img, bboxes, labels, masks
