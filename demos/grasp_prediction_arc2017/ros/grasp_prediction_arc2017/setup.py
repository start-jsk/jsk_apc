from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup


d = generate_distutils_setup(
    packages=['grasp_prediction_arc2017_lib'],
    package_dir={'': 'python'},
)

setup(**d)
