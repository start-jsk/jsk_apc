#!/usr/bin/env python

import os.path as osp

import jsk_data


def main():
    jsk_data.download_data(
        pkg_name='baxtergv6_apc2016',
        path='data/objects.zip',
        url='https://drive.google.com/uc?id=1wnk81kntYpR0O3-gTfLCCvXkdA-R0_BA',
        md5='8ac8943470acdf3db8e59723bf6c4d67',
        extract=True,
    )

    jsk_data.download_data(
        pkg_name='baxtergv6_apc2016',
        path='data/models/fcn32s_v2_148000.npz',
        url='https://drive.google.com/uc?id=1s55y_dRVmswS2bsx2wb9wKTgd-OyHc2w',
        md5='5f8e439543fe1d69e003bede22e60472',
    )


if __name__ == '__main__':
    main()
