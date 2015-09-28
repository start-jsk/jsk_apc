#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['jsk_2015_apc_common'],
    package_dir={'': 'src'},
)

setup(**d)
