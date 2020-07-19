#!/usr/bin/env python

import argparse
import multiprocessing

import jsk_data


def download_data(*args, **kwargs):
    p = multiprocessing.Process(
        target=jsk_data.download_data,
        args=args,
        kwargs=kwargs)
    p.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='quiet', action='store_false')
    args = parser.parse_args()
    quiet = args.quiet

    PKG = 'dualarm_grasping'

    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/human_anno/20171129_iter00200000.npz',
        url='https://drive.google.com/uc?id=1EDoJMJukap71UrCg1Sm29vuDDqhWkqwd',
        md5='a7cbb66a2f37bcb04955286f6d92662c',
        quiet=quiet
    )
    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/self_anno/20180202_iter00200000.npz',
        url='https://drive.google.com/uc?id=1eOBvwALy-R-SBWKL6GYNTGiGzlBeBsV4',
        md5='e246250eaf51c45b90c5f8fed9d29286',
        quiet=quiet
    )

    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/self_anno/201802220906_iter00200000.npz',
        url='https://drive.google.com/uc?id=1dKXj2TzryyRoW7aRHZJeG6Gkl62GGB-r',
        md5='032abfbdc26bcad7b4d933c202e3e91b',
        quiet=quiet
    )

    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/human_anno/201802220907_iter00200000.npz',
        url='https://drive.google.com/uc?id=1Fi3-adAMq_FL12KSmfLfGE-3SbxGeLip',
        md5='35904d072c1174d110b5b6d7c9a65659',
        quiet=quiet
    )

    # trained with 1st sampling data
    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/self_anno/201802241311_iter00012000.npz',
        url='https://drive.google.com/uc?id=1H5yQ8zGglaoIdZ4f1CS8yAAY--i5Vj44',
        md5='92c0a37021004bf5ad52e42a793e4a7a',
        quiet=quiet
    )

    # trained with 2nd sampling data
    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/self_anno/201802261200_iter00012000.npz',
        url='https://drive.google.com/uc?id=1yGdgp6Bvw2Omn57Cd6517MVhox-0RTPL',
        md5='bd6daa72a15801e382202c591c9ad8c1',
        quiet=quiet
    )

    # trained with 2nd sampling data
    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/self_anno/201802261200_iter00048000.npz',
        url='https://drive.google.com/uc?id=1N1YvzSEXbLgTxDZMNd-el_3fbryx253j',
        md5='bb6b14f261dbbe0f3b4011ca626d3de3',
        quiet=quiet
    )

    # trained with 2nd sampling data
    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/self_anno/201802261200_iter00200000.npz',
        url='https://drive.google.com/uc?id=1xIf4lp1T4lpVadIRmfiLdc76clKYIBG2',
        md5='d80aa45ab04c6fd66e15ea819699296b',
        quiet=quiet
    )

    # trained with 2nd sampling data
    download_data(
        pkg_name=PKG,
        path='models/dualarm_grasp/self_anno/201802261705_iter00012000.npz',
        url='https://drive.google.com/uc?id=1UzABg0E3-aoO73wJHQSEf4VH4fAoRSp_',
        md5='6933e9341ff8eb16b33ed8b1c36e690b',
        quiet=quiet
    )

    # mask rcnn
    # trained with synthesized data
    download_data(
        pkg_name=PKG,
        path='models/dualarm_occluded_grasp/self_anno/20181127_model_iter_54480.npz',  # NOQA
        url='https://drive.google.com/uc?id=1ITLgC8TMHRljfq76hacQJ0FpMM3aSPWV',
        md5='53ed87ab425b65fd02bd47109dc505f9',
        quiet=quiet
    )
    download_data(
        pkg_name=PKG,
        path='models/dualarm_occluded_grasp/self_anno/20181127_params.yaml',  # NOQA
        url='https://drive.google.com/uc?id=1REdW1rRAKtSCzyVTbckvs7yzjO-YsD1n',
        md5='e3fd1169da238a96708cbfea8cc3c1fa',
        quiet=quiet
    )

    # # mask rcnn
    # # trained with 1st sampling
    # download_data(
    #     pkg_name=PKG,
    #     path='models/dualarm_occluded_grasp/self_anno/20181218_model_iter_26568.npz',  # NOQA
    #     url='https://drive.google.com/uc?id=1hhSi8RUXNq91y7UbEYrXlmvGJLc9uakJ',
    #     md5='7d2b9dfc605c36cb3de1047ab20197fe',
    #     quiet=quiet
    # )
    # download_data(
    #     pkg_name=PKG,
    #     path='models/dualarm_occluded_grasp/self_anno/20181218_params.yaml',  # NOQA
    #     url='https://drive.google.com/uc?id=12bSetNJhBTYCdIJf5GNbP2RHKr0nnM7r',
    #     md5='68682e89f414865b5d0e9557995610a8',
    #     quiet=quiet
    # )

    # mask rcnn
    # trained with 1st sampling
    download_data(
        pkg_name=PKG,
        path='models/dualarm_occluded_grasp/self_anno/20181226_model_iter_13284.npz',  # NOQA
        url='https://drive.google.com/uc?id=1jhzj5D0iAJgBwRLtQCZTLYk6rl-MbGuc',
        md5='05514904be0c18d815ff98e7cd8e0b1b',
        quiet=quiet
    )
    download_data(
        pkg_name=PKG,
        path='models/dualarm_occluded_grasp/self_anno/20181226_params.yaml',  # NOQA
        url='https://drive.google.com/uc?id=15CLY2OnFnk_LvDPKLlx1uqofqxmvmhCm',
        md5='90feeabaa144085c712b223d517b930f',
        quiet=quiet
    )

    # # mask rcnn
    # # trained with 2nd sampling
    # download_data(
    #     pkg_name=PKG,
    #     path='models/dualarm_occluded_grasp/self_anno/20181226_213602_model_iter_13677.npz',  # NOQA
    #     url='https://drive.google.com/uc?id=1trtL2dmBt36FLw1m5efKwUGk9EapiIxa',
    #     md5='bb01a0704ab8733a135e7b492231d423',
    #     quiet=quiet
    # )
    # download_data(
    #     pkg_name=PKG,
    #     path='models/dualarm_occluded_grasp/self_anno/20181226_213602_params.yaml',  # NOQA
    #     url='https://drive.google.com/uc?id=1FQousbTcIQtjv-c0Svl80in8F5E4jzPF',
    #     md5='ff03abf9339612d2dba7fdcfe8743b27',
    #     quiet=quiet
    # )

    # mask rcnn
    # trained with 2nd sampling
    download_data(
        pkg_name=PKG,
        path='models/dualarm_occluded_grasp/self_anno/20181227_model_iter_4559.npz',  # NOQA
        url='https://drive.google.com/uc?id=1L_kMSDR0yryojL4iHvXUXtR0PKgQ8obl',
        md5='3951b47d0c2441b71a9069db3a1a1155',
        quiet=quiet
    )
    download_data(
        pkg_name=PKG,
        path='models/dualarm_occluded_grasp/self_anno/20181227_params.yaml',  # NOQA
        url='https://drive.google.com/uc?id=1uMtfdif6HXQdVVvGEBlStxqzVMmM4_u6',
        md5='23947a4dfc850a10679b845720e56f41',
        quiet=quiet
    )


if __name__ == '__main__':
    main()
