import os.path as osp
import shutil
import tempfile

import instance_occlsegm_lib


def test_extractall():
    tmp_dir = tempfile.mkdtemp()

    url = 'https://github.com/github/hub/archive/v2.12.8.zip'
    path = osp.join(tmp_dir, 'hub.zip')
    print('==> Downloading from {}'.format(url))
    instance_occlsegm_lib.data.download(url=url, path=path, quiet=True)

    print('==> Extracting {}'.format(path))
    files = instance_occlsegm_lib.data.extractall(path, to=osp.dirname(path))
    print(files)
    assert all(osp.isabs(f) and osp.exists(f) for f in files)
    assert isinstance(files, list)
    files = instance_occlsegm_lib.data.extractall(path)
    assert all(osp.isabs(f) and osp.exists(f) for f in files)
    assert isinstance(files, list)
    print('\n'.join(files))

    print('==> Removing {}'.format(tmp_dir))
    shutil.rmtree(tmp_dir)
