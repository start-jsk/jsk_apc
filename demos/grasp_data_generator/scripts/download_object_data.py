import os
import os.path as osp

from download_datasets import download


filepath = osp.dirname(osp.realpath(__file__))
datadir = osp.join(filepath, '../data/compressed')


def main():
    if not osp.exists(datadir):
        os.makedirs(datadir)
    download(
        url='https://drive.google.com/uc?id=1StwWbNYnPiFICr9S2qXAoyQxELmlHw1V',
        output=osp.join(datadir, 'ItemDataARC2017.zip'),
        md5sum='c8ad2268b7f2d16accd716c0269d4e5f',
        quiet=False)


if __name__ == '__main__':
    main()
