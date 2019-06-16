import os.path as osp
import shutil
import tempfile

import pytest

import grasp_fusion_lib


@pytest.mark.httpaccess
def test_download():
    tmp_dir = tempfile.mkdtemp()

    url = 'https://drive.google.com/uc?id=1SG1gC-g_URVnx0zNMW05_B6TSI88mdop'
    path = osp.join(tmp_dir, 'spam.txt')
    print('Downloading from {} for 3 times.'.format(url))
    for i in range(3):
        print('Downloading from {} {}/3'.format(url, i + 1))
        grasp_fusion_lib.data.download(
            url=url,
            path=path,
            md5='cb31a703b96c1ab2f80d164e9676fe7d',
        )

    print('File content of {}'.format(path))
    assert open(path).read() == 'spam\n'

    print('Removing {}'.format(tmp_dir))
    shutil.rmtree(tmp_dir)
