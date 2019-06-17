import numpy as np

from instance_occlsegm_lib.contrib import synthetic2d


class OcclusionSegmentationDataset(object):

    def __init__(self, split):
        assert split in ['train', 'test']
        data = synthetic2d.datasets.ARC2017OcclusionDataset(split)
        self._insta_data = data

        class_names = ['__background__'] + data.class_names.tolist()
        class_names = np.array(class_names)
        class_names.setflags(write=0)
        self.class_names = class_names

    def __len__(self):
        return len(self._insta_data)

    def __getitem__(self, i):
        img, bboxes, labels, masks = self._insta_data[i]

        labels += 1  # 0 represents background
        height, width = img.shape[:2]
        n_class = len(self.class_names)
        lbl_vis = np.zeros((height, width), dtype=np.int32)
        lbl_occ = np.zeros((height, width, n_class - 1), dtype=np.int32)
        for label, mask in zip(labels, masks):
            lbl_vis[mask == 1] = label
            lbl_occ[:, :, label - 1] = mask == 2

        return img, lbl_vis, lbl_occ


if __name__ == '__main__':
    from .utils import view_occlusion_segmentation_dataset
    data = OcclusionSegmentationDataset('train')
    data.split = 'train'
    view_occlusion_segmentation_dataset(data)
