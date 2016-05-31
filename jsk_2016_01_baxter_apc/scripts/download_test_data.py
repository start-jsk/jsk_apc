#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'jsk_2016_01_baxter_apc'

    download_data(
        pkg_name=PKG,
        path='test_data/2016-04-30-16-33-54_apc2016-bin-boxes.bag',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vZ2xLZG55OWNYTDQ',
        md5='79404ca882f3131361d323112443be42',
    )

    download_data(
        pkg_name=PKG,
        path='test_data/sib_kinect2.bag.tar.gz',
        url='https://drive.google.com/uc?id=0BzBTxmVQJTrGRERod3E5S3RxdE0',
        md5='c3aaaf507b48fc7022edd51bbe819e4d',
        extract=True,
    )

    download_data(
        pkg_name=PKG,
        path='test_data/sib_right_softkinetic.bag.tar.gz',
        url='https://drive.google.com/uc?id=0BzBTxmVQJTrGTmJEaTZ5bERhMzg',
        md5='f764107a2a2fda2bb4f800c519d97dc2',
        extract=True,
    )


if __name__ == '__main__':
    main()
