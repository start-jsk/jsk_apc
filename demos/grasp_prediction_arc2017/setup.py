import shlex
import subprocess

from setuptools import find_packages
from setuptools import setup


def git_hash():
    cmd = 'git log -1 --format="%h"'
    try:
        hash_ = subprocess.check_output(shlex.split(cmd)).decode().strip()
    except subprocess.CalledProcessError:
        hash_ = None
    return hash_


version = '0.1.2-0'


hash_ = git_hash()
if hash_ is not None:
    version = '%s.%s' % (version, hash_)


install_requires = []
with open('requirements.txt') as f:
    for req in f:
        if req.startswith('-e '):
            continue
        install_requires.append(req.strip())


setup(
    name='grasp_prediction_arc2017_lib',
    version=version,
    packages=find_packages(),
    install_requires=install_requires,
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='https://github.com/start-jsk/jsk_apc/tree/master/demos/grasp_prediction_arc2017',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
