import numpy as np

from grasp_fusion_lib.datasets.apc.apc2016 import JskAPC2016Dataset


def test_jsk_apc2016_dataset():
    for split in ['train', 'valid']:
        _test_jsk(JskAPC2016Dataset(split))


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
