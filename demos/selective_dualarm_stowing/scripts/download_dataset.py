#!/usr/bin/env python

import gdown
import os
import rospkg
import tarfile


def download_and_extract(url, path, extract_dir):
    gdown.download(url, path, False)
    tar = tarfile.open(path, 'r:gz')
    tar.extractall(path=extract_dir)
    tar.close()


def main():
    rospack = rospkg.RosPack()
    dataset_dir = os.path.join(
        rospack.get_path('selective_dualarm_stowing'), 'dataset')
    v1_dataset_path = os.path.join(
        dataset_dir, 'selective_dualarm_stowing_dataset_v1.tar.gz')
    v2_dataset_path = os.path.join(
        dataset_dir, 'selective_dualarm_stowing_dataset_v2.tar.gz')
    download_and_extract(
        'https://drive.google.com/open?id=0B5DV6gwLHtyJS2l3TElEMFdFRFU',
        v1_dataset_path,
        dataset_dir)
    download_and_extract(
        'https://drive.google.com/open?id=0B5DV6gwLHtyJQnlxNGtjV2UyWU0',
        v2_dataset_path,
        dataset_dir)

if __name__ == '__main__':
    main()
