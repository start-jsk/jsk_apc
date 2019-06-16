import hashlib
import os
import os.path as osp
try:
    import cPickle as pickle
except ImportError:
    import pickle
import warnings

import grasp_fusion_lib.data


memos = {}


def memoize(key=None):
    def _memoize(func):
        def func_wrapper(*args, **kwargs):
            if key:
                contents = pickle.dumps(key(*args, **kwargs))
            else:
                contents = pickle.dumps(
                    {'func': func.func_code.co_code,
                     'args': args, 'kwargs': kwargs})
            sha1 = hashlib.sha1(contents).hexdigest()
            if sha1 in memos:
                return memos[sha1]
            res = func(*args, **kwargs)
            if len(memos) > 50:
                memos.popitem()
            else:
                memos[sha1] = res
            return res
        return func_wrapper
    return _memoize


cache_root = osp.expanduser('~/.cache/grasp_fusion_lib/_func_cache')
try:
    os.makedirs(cache_root)
except OSError:
    if not osp.isdir(cache_root):
        raise


def cache(key=None):
    def _cache(func):
        def func_wrapper(*args, **kwargs):
            if key:
                contents = pickle.dumps(
                    {'key': key(*args, **kwargs),
                     'args': args, 'kwargs': kwargs})
            else:
                contents = pickle.dumps(
                    {'func': func.func_code.co_code,
                     'args': args, 'kwargs': kwargs})
            sha1 = hashlib.sha1(contents).hexdigest()
            cache_file = osp.join(cache_root, sha1)
            if osp.isfile(cache_file):
                warnings.warn('Loading from cache file: %s' % cache_file)
                return pickle.load(open(cache_file, 'rb'))
            res = func(*args, **kwargs)
            # 1GB
            if grasp_fusion_lib.data.get_directory_size(cache_root) > 1 * 1e9:
                warnings.warn('Cache size is over 1GB at: %s' % cache_root)
            pickle.dump(res, open(cache_file, 'wb'))
            return res
        return func_wrapper
    return _cache
