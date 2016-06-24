#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'jsk_apc2016_common'

    download_data(
        pkg_name=PKG,
        path='trained_data/vgg16_96000.chainermodel',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vOTdzOGlJcGM1N00',
        md5='3c993d333cf554684b5162c9f69b20cf',
    )


if __name__ == '__main__':
    main()
