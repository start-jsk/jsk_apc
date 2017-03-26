#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'jsk_arc2017_common'

    download_data(
        pkg_name=PKG,
        path='data/models/fcn32s_arc2017_dataset_v1_20170326_005.pth',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vT1pnWnVsNERHTVk',
        md5='ae9d13c126389bd63bccf0db1551f31e',
    )


if __name__ == '__main__':
    main()
