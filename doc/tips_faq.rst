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


Error related to machine tag in roslaunch.
------------------------------------------

If you have error like below, check `here <http://answers.ros.org/question/41446/a-is-not-in-your-ssh-known_hosts-file/>`_.

.. code-block:: bash

  % roslaunch jsk_2016_01_baxter_apc setup_torso.launch
  ... logging to /home/baxter/.ros/log/44aa3fbe-23c6-11e6-b5c0-000af716d1cb/roslaunch-sheeta-74117.log
  Checking log directory for disk usage. This may take awhile.
  Press Ctrl-C to interrupt
  Done checking log file disk usage. Usage is <1GB.

  started roslaunch server http://133.11.216.190:36416/
  remote[133.11.216.167-0] starting roslaunch
  remote[133.11.216.167-0]: creating ssh connection to 133.11.216.167:22, user[baxter]
  remote[133.11.216.167-0]: failed to launch on doura:

  133.11.216.167 is not in your SSH known_hosts file.

  Please manually:
    ssh baxter@133.11.216.167

  then try roslaunching again.

  If you wish to configure roslaunch to automatically recognize unknown
  hosts, please set the environment variable ROSLAUNCH_SSH_UNKNOWN=1
