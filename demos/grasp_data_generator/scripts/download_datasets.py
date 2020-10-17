import hashlib
import os
import os.path as osp

import gdown

filepath = osp.dirname(osp.realpath(__file__))
compressed_dir = osp.join(filepath, '../data/compressed')


def check_md5sum(output, md5sum):
    # validate md5 string length if it is specified
    if md5sum and len(md5sum) != 32:
        raise ValueError(
            'md5sum must be 32 charactors\n'
            'actual: {0} ({1} charactors)'.format(md5sum, len(md5sum)))
    print('[{0}] Checking md5sum ({1})'.format(output, md5sum))
    is_same = hashlib.md5(open(output, 'rb').read()).hexdigest() == md5sum
    print('[{0}] Finished checking md5sum: {1}'.format(output, is_same))
    return is_same


def download(url, output, md5sum, quiet=False):
    if not (osp.exists(output) and check_md5sum(output, md5sum)):
        gdown.download(url=url, output=output, quiet=quiet)
    else:
        print('[{0}] Skip downloading'.format(output))


def main():
    if not osp.exists(compressed_dir):
        os.makedirs(compressed_dir)
    # v1
    download(
        url='https://drive.google.com/uc?id=1H5Qz7FJBdyE1e1D2ZgA6j40ylat0Y9lG',
        output=osp.join(compressed_dir, 'dualarm_grasp_dataset_v1.tar.gz'),
        md5sum='0b5f06fbb3109deae38e5c6e719fbb45',
        quiet=False)
    # v2
    download(
        url='https://drive.google.com/uc?id=1lQiLODgmGxRFnf5U0XcaqtgMVvplvKcE',
        output=osp.join(compressed_dir, 'dualarm_grasp_dataset_v2.tar.gz'),
        md5sum='6bb8456cb36437792bf1fe62e6c34cb4',
        quiet=False)
    # v3
    download(
        url='https://drive.google.com/uc?id=1Xfjc5KiqgwE9vHTBMNBSeIvfCis04s4_',
        output=osp.join(compressed_dir, 'dualarm_grasp_dataset_v3.tar.gz'),
        md5sum='25e377e3a3f779279a0bbbf71cb14790',
        quiet=False)
    # v4
    download(
        url='https://drive.google.com/uc?id=1bvLjn-9ZckxiuRg6FImJEbUS56e_79kd',
        output=osp.join(compressed_dir, 'dualarm_grasp_dataset_v4.tar.gz'),
        md5sum='9674e8b4f96ebc6caae7b73436dda339',
        quiet=False)
    # v5
    download(
        url='https://drive.google.com/uc?id=1_CgyuWx-z3tKfAowjGfj4V2eey8wPb7I',
        output=osp.join(compressed_dir, 'dualarm_grasp_dataset_v5.tar.gz'),
        md5sum='ef9dafebfb8f133faa9dff2bf78163fb',
        quiet=False)
    # v6
    download(
        url='https://drive.google.com/uc?id=1qWuum_MldO_yeANF--8gskJUVmKZSEeA',
        output=osp.join(compressed_dir, 'dualarm_grasp_dataset_v6.tar.gz'),
        md5sum='87f6e46699b558d15b00b9d63b422aa0',
        quiet=False)
    # v7
    download(
        url='https://drive.google.com/uc?id=1LDIgG32PlQWTRxFOUpZCOYzW2_r1RTWd',
        output=osp.join(compressed_dir, 'dualarm_grasp_dataset_v7.tar.gz'),
        md5sum='6890c47d0ccf3c46eea18843692a4373',
        quiet=False)
    # v8
    download(
        url='https://drive.google.com/uc?id=1d-hqzqNBgjV5ifw408R4D_xCF97HO9dZ',
        output=osp.join(compressed_dir, 'dualarm_grasp_dataset_v8.tar.gz'),
        md5sum='528c0573480168da3c572e8ae3090911',
        quiet=False)

    # occluded instance
    # occluded instance v1
    download(
        url='https://drive.google.com/uc?id=1NUVCmt5bCsUeijEh_A6NdHhIsvyeqsrA',
        output=osp.join(
            compressed_dir,
            'occluded_instance_dualarm_grasp_dataset_v1.tar.gz'),
        md5sum='1707abf130b411c73ae5b55a77e6ef1a',
        quiet=False)

    # instance
    # instance v1
    download(
        url='https://drive.google.com/uc?id=1VZu7a6Adwp1a8ZYuiBKfHzAJi3sixOdq',
        output=osp.join(
            compressed_dir, 'instance_dualarm_grasp_dataset_v1.tar.gz'),
        md5sum='02d252e7dcf29dea56fda671cfb2cae8',
        quiet=False)

    # finetuning
    # fine tuning v1
    download(
        url='https://drive.google.com/uc?id=1812p5_kL4IjnCL32n8z-wkB_BOAedYYz',
        output=osp.join(
            compressed_dir, 'finetuning_dualarm_grasp_dataset_v1.tar.gz'),
        md5sum='a2fbd996498ed3574bd710cd14cbd20b',
        quiet=False)
    # finetuning v2
    download(
        url='https://drive.google.com/uc?id=1ZMPh2q5C-uAki8J4OCM5MhOtbEcVoBBx',
        output=osp.join(
            compressed_dir, 'finetuning_dualarm_grasp_dataset_v2.tar.gz'),
        md5sum='b45dced5bd402d75bc6471df74caefbe',
        quiet=False)
    # finetuning v3
    download(
        url='https://drive.google.com/uc?id=1_RUhaI46euw15PvZeMIcuxWQhjEv-t3i',
        output=osp.join(
            compressed_dir, 'finetuning_dualarm_grasp_dataset_v3.tar.gz'),
        md5sum='d345d6bfcaa4d5c42de3c5d78db265f9',
        quiet=False)
    # finetuning v4
    download(
        url='https://drive.google.com/uc?id=1KrSrL4VrqDRQdT6kTN4aTzHEmKPUNELg',
        output=osp.join(
            compressed_dir, 'finetuning_dualarm_grasp_dataset_v4.tar.gz'),
        md5sum='5a40def7700df542011247ad9d37f0cd',
        quiet=False)

    # occluded instance annotated dataset
    download(
        url='https://drive.google.com/uc?id=19VReD5ZzN-gyQgcdaCRTp-7zLl50hXyI',
        output=osp.join(
            compressed_dir, 'oi_real_annotated_dataset_v1.tar.gz'),
        md5sum='30fb688e8d1c1f9ebdacf5d7d9f19451',
        quiet=False)

    download(
        url='https://drive.google.com/uc?id=11fis7Q3xBfn6ny8ODi_jix-A4XuP57rp',
        output=osp.join(
            compressed_dir, 'oi_real_annotated_dataset_v2.tar.gz'),
        md5sum='ba65e043a791b6114cdcd5ac2fac166c',
        quiet=False)


if __name__ == '__main__':
    main()
