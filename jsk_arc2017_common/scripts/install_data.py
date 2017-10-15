#!/usr/bin/env python

import multiprocessing
import os

import jsk_data
import rospkg


PKG = 'jsk_arc2017_common'


try:
    rp = rospkg.RosPack()
    PKG_PATH = rp.get_path(PKG)
except rospkg.ResourceNotFound as e:
    print('ROS package [%s] is not found. '
          'Skipping by assuming we are in build.ros.org.' % PKG)
    quit(0)


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
    # dataset: v2 (improved)
    # augmentation: stack
    download_data(
        path='data/models/fcn32s_arc2017_datasetv2_cfg003_20170721.npz',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vNG5KcmxkWWN6Zk0',
        md5='980d3c6f0b3ef5e541f4db5280233c33',
    )

    # # dataset: v2
    # # augmentation: stack
    # # unknown_objects: apc2016
    # download_data(
    #     path='data/models/fcn32s_arc2017_datasetv2_cfg003_20170704.npz',
    #     url='https://drive.google.com/uc?id=0B9P1L--7Wd2vcEZMbVR0eHlVdDg',
    #     md5='606d2f6bc0c701e79b64466c73c83a72',
    # )

    # dataset: v3 (natural dataset)
    # augmentation: stack
    download_data(
        path='data/models/fcn32s_arc2017_datasetv3_cfg009_20170724.npz',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vWXRNa3BpeWlqcFE',
        md5='17536436262487185740e186e39651bf',
    )

    symlink_src = os.path.join(
        PKG_PATH, 'data/models/fcn32s_arc2017_datasetv3_cfg009_20170724.npz')
    symlink_dst = os.path.join(PKG_PATH, 'data/models/fcn32s.npz')
    if os.path.exists(symlink_dst):
        print('[%s] File already exists, so skipping.' % symlink_dst)
    else:
        if os.path.islink(symlink_dst):
            os.remove(symlink_dst)
        print('[%s] Creating symlink to: %s' % (symlink_dst, symlink_src))
        os.symlink(symlink_src, symlink_dst)


if __name__ == '__main__':
    main()
