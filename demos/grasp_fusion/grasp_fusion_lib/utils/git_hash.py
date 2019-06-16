import os.path as osp
import subprocess


def git_hash(filename=None):
    cwd = None
    if filename is not None:
        cwd = osp.dirname(osp.abspath(filename))
    cmd = 'git log -1 --format="%h"'
    return subprocess.check_output(cmd, shell=True, cwd=cwd).strip().decode()
