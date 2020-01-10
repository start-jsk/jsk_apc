#!/usr/bin/env python

import os.path as osp

import jsk_data


def main():
    path = osp.expanduser('~/data/hasegawa_iros2018_ros.zip')  # NOQA
    jsk_data.download_data(
        pkg_name='grasp_prediction_arc2017',
        path=path,
        url='https://drive.google.com/uc?id=1GmAyn1WECgZDSAria2NAHWWTNC4yVFsF',
        md5='fe6d539b08ae8825ce0258e16e363b9d',
        extract=True,
    )


if __name__ == '__main__':
    main()
