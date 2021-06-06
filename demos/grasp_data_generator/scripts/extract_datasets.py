import os
import os.path as osp
import tarfile


filepath = osp.dirname(osp.realpath(__file__))
compressed_dir = osp.join(filepath, '../data/compressed')
training_data_dir = osp.join(filepath, '../data/training_data')
finetuning_data_dir = osp.join(filepath, '../data/finetuning_data')
evaluation_data_dir = osp.join(filepath, '../data/evaluation_data')


def extract(tarpath, extract):
    with tarfile.open(tarpath, 'r:gz') as tarf:
        tarf.extractall(path=extract)


def main():
    if not osp.exists(compressed_dir):
        os.makedirs(compressed_dir)
    if not osp.exists(training_data_dir):
        os.makedirs(training_data_dir)
    # v1
    extract(
        tarpath=osp.join(compressed_dir, 'dualarm_grasp_dataset_v1.tar.gz'),
        extract=training_data_dir)
    # v2
    extract(
        tarpath=osp.join(compressed_dir, 'dualarm_grasp_dataset_v2.tar.gz'),
        extract=training_data_dir)
    # v3
    extract(
        tarpath=osp.join(compressed_dir, 'dualarm_grasp_dataset_v3.tar.gz'),
        extract=training_data_dir)
    # v4
    extract(
        tarpath=osp.join(compressed_dir, 'dualarm_grasp_dataset_v4.tar.gz'),
        extract=training_data_dir)
    # v5
    extract(
        tarpath=osp.join(compressed_dir, 'dualarm_grasp_dataset_v5.tar.gz'),
        extract=training_data_dir)
    # v6
    extract(
        tarpath=osp.join(compressed_dir, 'dualarm_grasp_dataset_v6.tar.gz'),
        extract=training_data_dir)
    # v7
    extract(
        tarpath=osp.join(compressed_dir, 'dualarm_grasp_dataset_v7.tar.gz'),
        extract=training_data_dir)
    # v8
    extract(
        tarpath=osp.join(compressed_dir, 'dualarm_grasp_dataset_v8.tar.gz'),
        extract=training_data_dir)

    # instance v1
    extract(
        tarpath=osp.join(
            compressed_dir, 'instance_dualarm_grasp_dataset_v1.tar.gz'),
        extract=training_data_dir)

    # occluded instance v1
    extract(
        tarpath=osp.join(
            compressed_dir,
            'occluded_instance_dualarm_grasp_dataset_v1.tar.gz'),
        extract=training_data_dir)

    # finetuning v1
    extract(
        tarpath=osp.join(
            compressed_dir, 'finetuning_dualarm_grasp_dataset_v1.tar.gz'),
        extract=finetuning_data_dir)
    # finetuning v2
    extract(
        tarpath=osp.join(
            compressed_dir, 'finetuning_dualarm_grasp_dataset_v2.tar.gz'),
        extract=finetuning_data_dir)
    # finetuning v3
    extract(
        tarpath=osp.join(
            compressed_dir, 'finetuning_dualarm_grasp_dataset_v3.tar.gz'),
        extract=finetuning_data_dir)
    # finetuning v4
    extract(
        tarpath=osp.join(
            compressed_dir, 'finetuning_dualarm_grasp_dataset_v4.tar.gz'),
        extract=finetuning_data_dir)

    # evaluation v1
    extract(
        tarpath=osp.join(
            compressed_dir, 'oi_real_annotated_dataset_v1.tar.gz'),
        extract=evaluation_data_dir)

    # evaluation v2
    extract(
        tarpath=osp.join(
            compressed_dir, 'oi_real_annotated_dataset_v2.tar.gz'),
        extract=evaluation_data_dir)


if __name__ == '__main__':
    main()
