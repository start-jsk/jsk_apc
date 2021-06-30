import os
import re
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    init_py = open(os.path.join(here, 'concarne', '__init__.py')).read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''
try:
    # obtain long description from README and CHANGES
    README = open(os.path.join(here, 'README')).read()
    #CHANGES = open(os.path.join(here, 'CHANGES.rst')).read()
except IOError:
    README = ''
    #CHANGES = ''

install_requires = [
    'numpy',
    'scipy',
    'matplotlib',
    'scikit-learn>=0.16'
    ]

setup(name='rbo-apc-object-recognition',
    version=version,
    description='Code accompanying our paper "Probabilistic Object Segmentation for the Amazon Picking Challenge"',
    long_description=README,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author='Rico Jonschkowski, Sebastian Hoefer',
    author_email='rico.jonschkowski@tu-berlin.de, sebastian.hoefer@tu-berlin.de',
    url='https://gitlab.tubit.tu-berlin.de/rbo-lab/rbo-apc-object-segmentation',
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    )


