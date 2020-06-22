#!/usr/bin/env python

import os.path as osp

import jsk_data


def main():
    path = osp.expanduser('~/data/hasegawa_mthesis_ros.zip')  # NOQA
    jsk_data.download_data(
        pkg_name='grasp_prediction_arc2017',
        path=path,
        url='https://drive.google.com/uc?id=1UKqSrHFm3ZihLfeXPVxUT5pa4FyenkQo',
        md5='38bd2d228c715dd05854c293e5e453a9',
        extract=True,
    )


if __name__ == '__main__':
    main()
