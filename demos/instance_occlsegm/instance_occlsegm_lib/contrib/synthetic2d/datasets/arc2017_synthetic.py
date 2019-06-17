import os.path as osp

import chainer
import numpy as np

import instance_occlsegm_lib.data
from instance_occlsegm_lib.datasets.apc.arc2017.jsk import ItemDataDataset

from instance_occlsegm_lib.datasets.apc.arc2017_v2 import load_item_data
from instance_occlsegm_lib.datasets.apc import \
    ARC2017ItemDataSyntheticInstanceSegmentationDataset


class ARC2017SyntheticDataset(ItemDataDataset):

    def __init__(self, do_aug=False, aug_level='all'):
        if not osp.exists(self.item_data_dir):
            self.download()

        ret_load_item_data = load_item_data(self.item_data_dir)

        super(ARC2017SyntheticDataset, self).__init__(
            split='train',
            ret_load_item_data=ret_load_item_data,
            do_aug=do_aug,
            aug_level=aug_level,
            from_scratch=True,
            skip_known=False,
            verbose=False,
        )

    def __len__(self):
        return int(10e3)

    #     cls.item_data_dir = \
    #         osp.expanduser('~/data/arc2017/datasets/ItemDataARC2017')

    # # In order to acquire ItemDataARC2017_MaskPred from scratch,
    # #   1. you first run this to download raw item data.
    # #   2. you run generate_item_data_mask_pred.py next.
    # @classmethod
    # def download(cls):
    #     instance_occlsegm_lib.data.download(
    #         url='https://drive.google.com/uc?id=1hJe4JZvqc2Ni1sjuKwXuBxgddHH2zNFa',  # NOQA
    #         md5='c8ad2268b7f2d16accd716c0269d4e5f',
    #         path=cls.item_data_dir + '.zip',
    #         postprocess=instance_occlsegm_lib.data.extractall,
    #     )

    item_data_dir = \
        osp.expanduser('~/data/instance_occlsegm_lib/synthetic2d/datasets/ItemDataARC2017_MaskPred')  # NOQA

    @classmethod
    def download(cls):
        instance_occlsegm_lib.data.download(
            url='https://drive.google.com/uc?id=1OYoLwsRuHKP8is-7JIE2ii5f-VeICjv1',  # NOQA
            md5='5a64bb03613589e5aeab41d8319bb945',
            path=cls.item_data_dir + '.zip',
            postprocess=instance_occlsegm_lib.data.extractall,
        )


class ARC2017SyntheticInstancesDataset(
    ARC2017ItemDataSyntheticInstanceSegmentationDataset
):

    def __init__(self, do_aug=False, aug_level='all'):
        item_data_dir = ARC2017SyntheticDataset.item_data_dir
        if not osp.exists(item_data_dir):
            ARC2017SyntheticDataset.download()
        super(ARC2017SyntheticInstancesDataset, self).__init__(
            item_data_dir, do_aug=do_aug, aug_level=aug_level
        )


# -----------------------------------------------------------------------------
# Deprecated


class ARC2017SyntheticCachedDataset(chainer.dataset.DatasetMixin):

    class_names = None
    dataset_dir = osp.expanduser('~/data/instance_occlsegm_lib/synthetic2d/ARC2017SyntheticCachedDataset')  # NOQA

    def __init__(self, split):
        assert split in ['train', 'test']
        self._split = split
        self.class_names = ARC2017SyntheticDataset().class_names

        size_all = int(10e3)
        size_train = int(size_all * 0.75)
        size_test = size_all - size_train
        self._size = dict(train=size_train, test=size_test)

    def __len__(self):
        return self._size[self._split]

    def get_example(self, i):
        if self._split == 'train':
            j = i
        else:
            assert self._split == 'test'
            j = i + self._size['train']

        cache_file = osp.join(self.dataset_dir, '%08d.npz' % j)
        cache_data = np.load(cache_file)
        return cache_data['img'], cache_data['lbl']
