import os.path as osp

import chainer
import yaml

from ..apc2016 import class_names_apc2016


here = osp.dirname(osp.abspath(__file__))


DATASETS_DIR = osp.expanduser('~/data/arc2017/datasets')
with open(osp.join(here, 'data/object_names.yaml')) as f:
    class_names_arc2017 = yaml.safe_load(f)
class_names_arc2017 = tuple(class_names_arc2017)


def get_class_id_map_from_2016_to_2017():
    cls_names_16 = class_names_apc2016
    cls_names_17 = class_names_arc2017

    cls_name_16_to_17 = {
        'background': '__shelf__',
        'womens_knit_gloves': 'black_fashion_gloves',
        'crayola_24_ct': 'crayons',
        'scotch_duct_tape': 'duct_tape',
        'expo_dry_erase_board_eraser': 'expo_eraser',
        'hanes_tube_socks': 'hanes_socks',
        'laugh_out_loud_joke_book': 'laugh_out_loud_jokes',
        'rolodex_jumbo_pencil_cup': 'mesh_cup',
        'ticonderoga_12_pencils': 'ticonderoga_pencils',
        'kleenex_tissue_box': 'tissue_box',
    }
    for cls_nm in cls_names_16:
        if cls_nm not in cls_name_16_to_17:
            cls_name_16_to_17[cls_nm] = 'unknown'

    print('{:>28} -> {:<15}'.format('apc2016', 'arc2017'))
    print('-' * 53)
    cls_id_16_to_17 = {}
    for n16, n17 in cls_name_16_to_17.items():
        assert n16 in cls_names_16
        assert n17 in cls_names_17
        print('{:>28} -> {:<15}'.format(n16, n17))
        cls_id_16_to_17[cls_names_16.index(n16)] = cls_names_17.index(n17)

    return cls_id_16_to_17


class ARC2017NoUnknownDataset(chainer.dataset.DatasetMixin):

    class_names = list(class_names_arc2017)
    class_names[0] = '__background__'  # unknown -> __background__
    class_names = class_names[:-1]  # remove __shelf__

    def __init__(self, dataset):
        self._dataset = dataset
        self.split = dataset.split

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        img, lbl = self._dataset.get_example(i)
        lbl[lbl == 0] = -1  # ignore unknown label
        lbl[lbl == 41] = 0  # set __shelf__ to background
        return img, lbl
