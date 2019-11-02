import os.path as osp
import shutil
import tempfile

import grasp_fusion_lib


def test_extractall():
    tmp_dir = tempfile.mkdtemp()

    url = 'https://drive.google.com/uc?id=1qmgz4dY1i_2NiAEgmsPRKNc7K7u-9WRa'  # NOQA
    md5 = '9ad4c3db5b52178819c083582f0a7f87'
    path = osp.join(tmp_dir, 'VOCdevkit_18-May-2011.tar')
    print('==> Downloading from {}'.format(url))
    grasp_fusion_lib.data.download(url=url, path=path, md5=md5, quiet=True)

    print('==> Extracting {}'.format(path))
    files = grasp_fusion_lib.data.extractall(path, to=osp.dirname(path))
    print(files)
    assert all(osp.isabs(f) and osp.exists(f) for f in files)
    assert isinstance(files, list)
    files = grasp_fusion_lib.data.extractall(path)
    assert all(osp.isabs(f) and osp.exists(f) for f in files)
    assert isinstance(files, list)
    print('\n'.join(files))

    print('==> Removing {}'.format(tmp_dir))
    shutil.rmtree(tmp_dir)
