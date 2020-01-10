#!/usr/bin/env python

import os.path as osp

import jsk_data


def main():
    path = osp.expanduser('~/data/arc2017/system_inputs_jsons/pick_re-experiment.zip')  # NOQA
    jsk_data.download_data(
        pkg_name='jsk_arc2017_common',
        path=path,
        url='https://drive.google.com/uc?id=16ebONejvSC3j6Zp-6nAjQJqT-VLS7j5X',
        md5='55e64309cf88bde3752874e8f6d6d60d',
        extract=True,
    )


if __name__ == '__main__':
    main()
