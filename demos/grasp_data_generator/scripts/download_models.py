import os
import os.path as osp

from download_datasets import download


filepath = osp.dirname(osp.realpath(__file__))
datadir = osp.join(filepath, '../data/models')


def main():
    if not osp.exists(datadir):
        os.makedirs(datadir)
    download(
        url='https://drive.google.com/uc?id=1k7qCONNta6WDjqdXAXdem8rDQjWPWeFb',
        output=osp.join(datadir, 'model_00010000.npz'),
        md5sum='5739cb23249993428dcc67e6d763b00a',
        quiet=False)


if __name__ == '__main__':
    main()
