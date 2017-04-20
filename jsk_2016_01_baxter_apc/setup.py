from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['jsk_2016_01_baxter_apc'],
    package_dir={'': 'src'}
)

setup(**d)
