from __future__ import print_function

import hashlib
import os
import os.path as osp
try:
    import cPickle as pickle
except ImportError:
    import pickle
import shutil
import sys
import tarfile
import tempfile
import warnings
import zipfile

import filelock
import gdown


cache_root = osp.expanduser('~/.cache/grasp_fusion_lib/_func_cache')
try:
    os.makedirs(cache_root)
except OSError:
    if not osp.isdir(cache_root):
        raise


def test_md5(path, md5, quiet=False):
    """Check file identity with MD5.

    Parameters
    ----------
    path: str
        File path.
    md5: str
        Expected file hash. It must be 32 characters.
    """
    if not (isinstance(md5, str) and len(md5) == 32):
        raise ValueError('MD5 must be 32 characters.')
    if not quiet:
        print('[%s] Computing md5. Expected: %s' % (path, md5))
    md5_actual = hashlib.md5(open(path, 'rb').read()).hexdigest()
    if not quiet:
        print('[%s] Computed md5. Expected: %s, Actual: %s' %
              (path, md5, md5_actual))
    return md5_actual == md5


def download(url, path=None, md5=None, postprocess=None, quiet=False):
    """Download file from URL.

    Parameters
    ----------
    url: str
        URL from where the file is downloaded.
    path: str
        Path where file is placed.
    md5: str or None
        If None, the file existence is only checked.
    postprocess: function or None
        If not None, it is called after the download with path as the argument.
    quiet: bool
        If True, no message is printed while downloading.

    Returns
    -------
    None

    """
    if path is None:
        path = url.replace('/', '-SLASH-')\
                  .replace(':', '-COLON-')\
                  .replace('=', '-EQUAL-')\
                  .replace('?', '-QUESTION-')
        path = osp.join(cache_root, path)

    # check existence
    if osp.exists(path) and not md5:
        if not quiet:
            print('[%s] File exists.' % path)
        return path
    elif osp.exists(path) and md5 and test_md5(path, md5, quiet=quiet):
        return path

    # download
    lock_path = osp.join(cache_root, '_dl_lock')
    try:
        os.makedirs(osp.dirname(path))
    except OSError:
        pass
    temp_root = tempfile.mkdtemp(dir=cache_root)
    try:
        temp_path = osp.join(temp_root, 'dl')
        gdown.download(url, temp_path, quiet=quiet)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, path)
        if not quiet:
            print('Saved to: {}'.format(path), file=sys.stderr)
    except Exception:
        shutil.rmtree(temp_root)
        raise

    # postprocess
    if postprocess is not None:
        postprocess(path)

    return path


def extract(path, to='.', quiet=False):
    warnings.warn('Function extract is deprecated. Please use extractall.')
    del quiet
    return extractall(path, to=to)


def extractall(path, to=None):
    """Extract archive file.

    Parameters
    ----------
    path: str
        Path of archive file to be extracted.
    to: str, optional
        Directory to which the archive file will be extracted.
        If None, it will be set to the parent directory of the archive file.
    """
    if to is None:
        to = osp.dirname(path)

    list_func = lambda x: [m.path for m in x.members]
    if path.endswith('.zip'):
        opener, mode, list_func = zipfile.ZipFile, 'r', lambda x: x.namelist()
    elif path.endswith('.tar'):
        opener, mode = tarfile.open, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else:
        raise ValueError("Could not extract '%s' as no appropriate "
                         "extractor is found" % path)

    def list_func_abs(f):
        files = []
        for fname in list_func(f):
            fname = osp.join(to, fname)
            files.append(fname)
        return files

    with opener(path, mode) as f:
        f.extractall(path=to)

    return list_func_abs(f)


def directory_to_hash(dirpath):
    dirpath = osp.realpath(dirpath)
    info = []
    for dpath, dnames, fnames in os.walk(dirpath):
        info.append(os.stat(dpath))
        for dname in dnames:
            dname = osp.join(dpath, dname)
            info.append((dname, os.stat(dname)))
        for fname in fnames:
            fname = osp.join(dpath, fname)
            info.append((fname, os.stat(fname)))
    return hashlib.sha256(pickle.dumps(info)).hexdigest()


def get_directory_size(dirpath):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dirpath):
        for f in filenames:
            fp = osp.join(dirpath, f)
            total_size += osp.getsize(fp)
    return total_size
