#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'jsk_arc2017_common'

    # augmentation: standard
    download_data(
        pkg_name=PKG,
        path='data/models/fcn32s_arc2017_dataset_v1_20170326_005.pth',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vT1pnWnVsNERHTVk',
        md5='ae9d13c126389bd63bccf0db1551f31e',
    )

    # augmentation: stack
    download_data(
        pkg_name=PKG,
        path='data/models/fcn32s_arc2017_dataset_v1_20170417.pth',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vYWloN0FGeEhlcGs',
        md5='a098399a456de29ef8d4feaa8ae795e9',
    )


if __name__ == '__main__':
    main()
