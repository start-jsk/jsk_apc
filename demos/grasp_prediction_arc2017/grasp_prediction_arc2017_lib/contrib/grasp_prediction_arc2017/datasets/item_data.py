import os.path as osp

import grasp_fusion_lib


def pick_re_experiment():
    path = osp.expanduser('~/data/arc2017/item_data/pick_re-experiment.zip')
    grasp_fusion_lib.data.download(
        url='https://drive.google.com/uc?id=1yv4kjk4ZmNceiW9-04qQf6mobbV0qsRZ',
        path=path,
        md5='e83bf4dd073f9b93ed124b2b09b84486',
        postprocess=grasp_fusion_lib.data.extractall,
    )
    return osp.splitext(path)[0]


def hasegawa_iros2018():
    path = osp.expanduser(
        '~/data/hasegawa_iros2018/item_data/ItemDataBooks6.zip')
    grasp_fusion_lib.data.download(
        url='https://drive.google.com/uc?id=15ADXORIyQr9X8rpM38aqtVLCYlzAg7qz',
        path=path,
        md5='d65a586726198f4c94b2080c9c21a6c4',
        postprocess=grasp_fusion_lib.data.extractall,
    )
    return osp.splitext(path)[0]


def hasegawa_master_thesis():
    path = osp.expanduser(
        '~/data/hasegawa_master_thesis/item_data/ItemDataBooks8.zip')
    grasp_fusion_lib.data.download(
        url='https://drive.google.com/uc?id=1Tln2FBNct6OzT-uZhL3YGQ3Y_eLD41xh',
        path=path,
        md5='b5fa2047790cc247593f022c371f12ae',
        postprocess=grasp_fusion_lib.data.extractall,
    )
    return osp.splitext(path)[0]
