#!/usr/bin/env python

import multiprocessing

import jsk_data


PKG = 'jsk_arc2017_common'


def download_data(path, url, md5):
    p = multiprocessing.Process(
        target=jsk_data.download_data,
        kwargs=dict(
            pkg_name=PKG,
            path=path,
            url=url,
            md5=md5,
        ),
    )
    p.start()


def main():
    # # dataset: v1
    # # augmentation: standard
    # download_data(
    #     path='data/models/fcn32s_arc2017_dataset_v1_20170326_005.pth',
    #     url='https://drive.google.com/uc?id=0B9P1L--7Wd2vT1pnWnVsNERHTVk',
    #     md5='ae9d13c126389bd63bccf0db1551f31e',
    # )

    # # dataset: v1
    # # augmentation: stack
    # download_data(
    #     path='data/models/fcn32s_arc2017_dataset_v1_20170417.pth',
    #     url='https://drive.google.com/uc?id=0B9P1L--7Wd2vYWloN0FGeEhlcGs',
    #     md5='a098399a456de29ef8d4feaa8ae795e9',
    # )

    # dataset: v2
    # augmentation: stack
    download_data(
        path='data/models/fcn32s_arc2017_datasetv2_cfg003_20170612.npz',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vS1VaWWVFNDVFQ1k',
        md5='e4e07b66ebeaf6b33a79eb1b605ee3a3',
    )

    # dataset: v2
    # augmentation: stack
    # unknown_objects: apc2016
    download_data(
        path='data/models/fcn32s_arc2017_datasetv2_cfg003_20170704.npz',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vcEZMbVR0eHlVdDg',
        md5='606d2f6bc0c701e79b64466c73c83a72',
    )


if __name__ == '__main__':
    main()
