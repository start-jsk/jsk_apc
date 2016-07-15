#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'jsk_apc2016_common'

    download_data(
        pkg_name=PKG,
        path='data/sib_rbo_tokyo.zip',
        url='https://drive.google.com/uc?id=0BzBTxmVQJTrGdE9QRWFrdHV1MjA',
        md5='4e8e54855e184a193fb55cfbead6f1b6',
        extract=True)


if __name__ == '__main__':
    main()
