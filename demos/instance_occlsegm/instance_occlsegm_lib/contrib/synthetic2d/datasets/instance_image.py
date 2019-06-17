import os.path as osp

import numpy as np

import instance_occlsegm_lib.data
from instance_occlsegm_lib.datasets.apc.arc2017_v2 import load_item_data


def _download(url, path, md5):
    instance_occlsegm_lib.data.download(
        url=url,
        path=path,
        md5=md5,
        postprocess=instance_occlsegm_lib.data.extractall,
    )
    return osp.splitext(path)[0]


class InstanceImageDataset(object):

    class_names = np.array(['bg', 'fg'])

    item_data_dirs = {
        'arc2017_all': {
            'url': 'https://drive.google.com/uc?id=1RS3C6MUmlSwq8OP3gU-SaIBB-Uh9U3w4',  # NOQA
            'path': osp.expanduser('~/data/arc2017/datasets/ItemDataAll.zip'),  # NOQA
            'md5': 'a55f0e7e66d74e5300b087bcfdd44242',
        },
        'arc2017': {
            'url': 'https://drive.google.com/uc?id=1hJe4JZvqc2Ni1sjuKwXuBxgddHH2zNFa',  # NOQA
            'path': osp.expanduser('~/data/arc2017/datasets/ItemDataARC2017.zip'),  # NOQA
            'md5': None,
        },
    }

    def __init__(self, item_data_dir='arc2017_all', load_pred=True):
        if item_data_dir in self.item_data_dirs:
            entry = self.item_data_dirs[item_data_dir]
            item_data_dir = _download(**entry)
        self.item_data_dir = item_data_dir
        object_names, object_data = load_item_data(
            self.item_data_dir, load_pred=load_pred
        )
        self.object_data = object_data

    def __len__(self):
        return len(self.object_data)

    def __getitem__(self, index):
        obj_datum = self.object_data[index]
        img = obj_datum['img']
        lbl_fgbg = obj_datum['mask'].astype(np.int32)
        return img, lbl_fgbg
