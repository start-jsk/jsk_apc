import os.path as osp
import shutil
import tempfile

import grasp_fusion_lib


def test_extractall():
    tmp_dir = tempfile.mkdtemp()

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar'  # NOQA
    path = osp.join(tmp_dir, 'VOCdevkit_18-May-2011.tar')
    print('==> Downloading from {}'.format(url))
    grasp_fusion_lib.data.download(url=url, path=path, quiet=True)

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
