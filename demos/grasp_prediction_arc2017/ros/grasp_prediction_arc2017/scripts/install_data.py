#!/usr/bin/env python

import os
import os.path as osp

import jsk_data


def main():
    path = osp.expanduser('~/data/grasp_prediction_arc2017/rosbags/arc2017_pick_red_composition_book.bag')  # NOQA
    jsk_data.download_data(
        pkg_name='grasp_prediction_arc2017',
        path=path,
        url='https://drive.google.com/uc?id=1u5W0VXNs_EjKF3MDpoaRTpuXkENAB7hy',
        md5='127747ec10ccb1c81688d88a28d58818',
    )

    path = osp.expanduser('~/data/grasp_prediction_arc2017/logs/fcn32s_CFG-000_VCS-2400e9e_TIME-20170827-233211/models/FCN32s_iter00044000.npz')  # NOQA
    jsk_data.download_data(
        pkg_name='grasp_prediction_arc2017',
        path=path,
        url='https://drive.google.com/uc?id=1y0i9Rt76-iFsFhKxJDMJbGDs_NvCalU5',
        md5='d2ace2f1b45f862b8976f81637a8e3a7',
    )

    path = osp.expanduser('~/data/hasegawa_iros2018/system_inputs/ForItemDataBooks6/FCN8sAtOnce_180720_130702_iter00060000.npz')  # NOQA
    jsk_data.download_data(
        pkg_name='grasp_prediction_arc2017',
        path=path,
        url='https://drive.google.com/uc?id=1ADYXvEouFL3ynnb8xBCmcAzgD8J-ddpm',
        md5='09fc1db2f5ff1a3f04366024e5921706',
    )

    path = osp.expanduser('~/data/hasegawa_iros2018/system_inputs/ForItemDataBooks6/objects.zip')  # NOQA
    jsk_data.download_data(
        pkg_name='grasp_prediction_arc2017',
        path=path,
        url='https://drive.google.com/uc?id=1dvC_QRtathHrg_PseCNkSiXwWmiT9IB_',
        md5='d25bff88acc16313210237da4f6367c4',
        extract=True,
    )


if __name__ == '__main__':
    main()
