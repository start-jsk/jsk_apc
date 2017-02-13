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

    download_data(
        pkg_name=PKG,
        path='trained_data/vgg16_rotation_translation_brightness_372000.chainermodel',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vamZrczFqb3JoVnM',
        md5='58a0e819ba141a34b1d68cc5e972615b',
    )

    download_data(
        pkg_name=PKG,
        path='trained_data/fcn32s_6000.chainermodel',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vd0JUMFZKeVQxaG8',
        md5='d063161d18129946f6c2878afb5f9067',
    )
    download_data(
        pkg_name=PKG,
        path='trained_data/fcn32s_v2_148000.chainermodel',
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vWHRGSnd4Q0ljTHc',
        md5='550182bacf34398b9bd72ab2939f06fd',
    )


if __name__ == '__main__':
    main()
