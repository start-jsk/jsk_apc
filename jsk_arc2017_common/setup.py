from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup


d = generate_distutils_setup(
    packages=['jsk_arc2017_common'],
    package_dir={'': 'python'},
)

setup(**d)
