Tips & FAQ
==========


Controlling joints of the robot does not work.
----------------------------------------------
Run below to synchronize the time with robot.
Time synchronization is crucial.

.. code-block:: bash

  sudo ntpdate baxter.jsk.imi.i.u-tokyo.ac.jp


Rosdep failure due to cython version.
-------------------------------------
.. code-block:: bash

  rosdep install -y -r --from-paths . --ignore-src

This command may fail with below errors.

.. code-block:: bash

  pkg_resources.DistributionNotFound: cython>=0.21
  ...
  ERROR: the following rosdeps failed to install
  pip: command [sudo -H pip install -U scikit-image] failed
  pip: Failed to detect successful installation of [scikit-image]

In this case, maybe your setuptools is too old. Please run below command.

.. code-block:: bash

  sudo pip install -U setuptools

https://github.com/start-jsk/jsk_apc/issues/1244 for details.


How to release a new version of jsk_apc?
----------------------------------------
.. code-block:: bash

  roscd jsk_apc
  catkin_generate_change_log
  # edit CHANGELOG.rst to create a pretty changelog
  catkin_prepare_release
  bloom-release --rosdistro indigo --track indigo jsk_apc  # you may need to fix package.xml for pip packages
