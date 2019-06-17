import glob
import os
import os.path as osp

import chainer
import numpy as np

import instance_occlsegm_lib
from instance_occlsegm_lib.contrib.synthetic2d.datasets.arc2017_occlusion \
    import _load_npz


def transform_to_panoptic(in_data, class_names):
    img, bboxes, labels, masks = in_data

    height, width = img.shape[:2]
    n_fg_class = len(class_names)  # fg_class_names
    lbl_vis = np.zeros((height, width), dtype=np.int32)
    lbl_occ = np.zeros((height, width, n_fg_class), dtype=np.int32)
    for label, mask in zip(labels, masks):
        lbl_vis[mask == 1] = label + 1
        lbl_occ[:, :, label] = mask == 2

    return img, bboxes, labels, masks, lbl_vis, lbl_occ


class PanopticOcclusionSegmentationDataset(chainer.datasets.TransformDataset):

    @staticmethod
    def download():
        url = 'https://drive.google.com/uc?id=1TlaUk9JDDaYAsqrg7TyKZNixKTX-ANIF'  # NOQA
        path = osp.expanduser('~/data/instance_occlsegm_lib/instance_occlsegm/dataset_data/20180204_splits.zip')  # NOQA
        instance_occlsegm_lib.data.download(
            url=url,
            path=path,
            md5='50450cf47afaa83ac0ebf7a9ce1ce43f',
            postprocess=instance_occlsegm_lib.data.extractall,
        )
        url = 'https://drive.google.com/uc?id=1K6GHGyT5uPhq2CbSyoJjt5xppsWXslj_'  # NOQA
        path = osp.expanduser('~/data/instance_occlsegm_lib/instance_occlsegm/dataset_data/20180730_splits.zip')  # NOQA
        instance_occlsegm_lib.data.download(
            url=url,
            path=path,
            md5='96fb4183510516bfe5d2977b829d2087',
            postprocess=instance_occlsegm_lib.data.extractall,
        )

    def __init__(self, split, augmentation=False):
        assert split in ['train', 'test']

        self._augmentation = augmentation

        dataset_dirs = ['20180204_splits', '20180730_splits']
        root_dir = osp.expanduser(
            '~/data/instance_occlsegm_lib/instance_occlsegm/dataset_data')

        dataset_dir = osp.join(root_dir, dataset_dirs[0])
        names_file = osp.join(dataset_dir, 'class_names.txt')
        if not osp.exists(names_file):
            self.download()
        fg_class_names = np.loadtxt(names_file, dtype=str)[1:]
        fg_class_names.setflags(write=0)
        self.class_names = fg_class_names

        video_dirs = []
        npz_files = []
        for dataset_dir in dataset_dirs:
            dataset_dir = osp.join(root_dir, dataset_dir)

            names_file = osp.join(dataset_dir, 'class_names.txt')
            if not osp.exists(names_file):
                self.download()
            assert (
                np.loadtxt(names_file, dtype=str)[1:] == fg_class_names
            ).all()

            split_dir = osp.join(dataset_dir, split)
            for video_dir in sorted(os.listdir(split_dir)):
                video_dir = osp.join(split_dir, video_dir)
                video_dirs.append(video_dir)
                npz_files.extend(
                    sorted(glob.glob(osp.join(video_dir, '*.npz')))
                )
        self._video_dirs = video_dirs
        self._npz_files = npz_files

    def __len__(self):
        return len(self._npz_files)

    def get_example(self, i):
        npz_file = self._npz_files[i]
        in_data = _load_npz(
            npz_file, augmentation=self._augmentation
        )
        return transform_to_panoptic(in_data, class_names=self.class_names)

    def get_video_datasets(self):
        datasets = []
        for video_dir in self._video_dirs:
            dataset = PanopticOcclusionSegmentationVideoDataset(
                video_dir, self.class_names, augmentation=self._augmentation
            )
            datasets.append(dataset)
        return datasets


class PanopticOcclusionSegmentationVideoDataset(
    chainer.datasets.TransformDataset
):

    def __init__(self, video_dir, class_names, augmentation=False):
        self.class_names = class_names
        self._npz_files = sorted(glob.glob(osp.join(video_dir, '*.npz')))
        self._augmentation = augmentation

    def __len__(self):
        return len(self._npz_files)

    def get_example(self, i):
        npz_file = self._npz_files[i]
        in_data = _load_npz(
            npz_file, augmentation=self._augmentation
        )
        return transform_to_panoptic(in_data, class_names=self.class_names)


if __name__ == '__main__':
    from .utils import view_panoptic_occlusion_segmentation_dataset
    data = PanopticOcclusionSegmentationDataset('train')
    data.split = 'train'
    # data = data.get_video_datasets()[0]
    view_panoptic_occlusion_segmentation_dataset(data)
