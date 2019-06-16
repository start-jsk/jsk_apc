#!/usr/bin/env python

import os.path as osp

import gdown


here = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(osp.expanduser('~'), 'data/grasp_fusion_lib/grasp_fusion')


def main():
    gdown.cached_download(
        url='https://drive.google.com/uc?id=1xZ-erCKOkzmyLldzw1RpsHzbXcEydkvT',
        path=osp.join(data_dir, 'rosbags/cluttered_tote.bag'),
        md5='1c9e45667d7a225d9bb089d2f45b3682',
    )

    gdown.cached_download(
        url='https://drive.google.com/uc?id=10bkcRYjwAu1z8MpigFFqIyUBcHk2MK4d',
        path=osp.join(data_dir, 'rosbags/cluttered_tote_for_suction_then_pinch.bag'),  # NOQA
        md5='14e1b4e6350acf33942841268213a9cf',
    )

    gdown.cached_download(
        url='https://drive.google.com/uc?id=1rsXuIL-CAhBAzsJvZ2ZbGrbIOhb2dTGk',
        path=osp.join(data_dir, 'models/instance_segmentation_model.npz'),
        md5='edc412932774b13cd4f20e2637377b5f',
    )


if __name__ == '__main__':
    main()
