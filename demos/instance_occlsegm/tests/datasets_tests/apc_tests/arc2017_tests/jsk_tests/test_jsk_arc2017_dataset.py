import itertools

import numpy as np

from instance_occlsegm_lib.datasets.apc.arc2017 import JskARC2017DatasetV1
from instance_occlsegm_lib.datasets.apc.arc2017 import JskARC2017DatasetV2
from instance_occlsegm_lib.datasets.apc.arc2017 import JskARC2017DatasetV3


def test_jsk_arc2017_dataset_aug():
    for split, aug in itertools.product(
        ['all', 'train', 'valid'], ['none', 'standard', 'stack']
    ):
        dataset = JskARC2017DatasetV3(split, aug=aug)
        _test_jsk(dataset)


def test_jsk_arc2017_dataset_noaug():
    for Dataset, split in itertools.product(
        [JskARC2017DatasetV1, JskARC2017DatasetV2], ['train', 'valid'],
    ):
        dataset = Dataset(split)
        _test_jsk(dataset)


def _test_jsk(dataset):
    assert len(dataset) > 0

    assert hasattr(dataset, 'split')
    assert hasattr(dataset, 'class_names')
    assert hasattr(dataset, '__getitem__')
    assert hasattr(dataset, '__len__')

    img, lbl = dataset[0]
    assert img.shape[:2] == lbl.shape

    n_class = len(dataset.class_names)
    assert np.all(np.unique(lbl) < n_class)
