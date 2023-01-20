#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'jsk_apc2016_common'

    download_data(
        pkg_name=PKG,
        path='samples/data/tabletop_objects.bag.tgz',
        url='https://drive.google.com/uc?id=1X4ORFGMA2zjdLbrKqb5FHtwIh0lXb9fz',
        md5='9047b0cfd31dfaef08b75204f64ae56f',
        extract=True,
    )


if __name__ == '__main__':
    main()
