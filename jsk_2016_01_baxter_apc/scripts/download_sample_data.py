#!/usr/bin/env python

import os
import os.path as osp

from jsk_data import download_data



def main():
    PKG = 'jsk_2016_01_baxter_apc'

    ros_home = osp.expanduser('~/.ros')
    data_dir = osp.join(ros_home, 'jsk_2016_01_baxter_apc')

    download_data(
        pkg_name=PKG,
        path=osp.join(data_dir,
                      'sample_eus_visualize_objs_2017-01-30-10-49-18.bag'),
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vVFZJRG9NQXBGVEk',
        md5='c60aa88c71e0b857940a53ed17b3653f',
    )


if __name__ == '__main__':
    main()
