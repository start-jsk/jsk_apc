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
