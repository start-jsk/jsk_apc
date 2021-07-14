^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package grasp_fusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.3.0 (2021-07-14)
------------------
* Merge pull request `#2751 <https://github.com/start-jsk/jsk_apc/issues/2751>`_ from knorth55/fix-version
  fix version in demo packages
* fix version in demo packages
  change to 4.2.1 for all other jsk_apc packages
* Merge branch 'master' into add-sleep
* Merge pull request `#2723 <https://github.com/start-jsk/jsk_apc/issues/2723>`_ from knorth55/diable-venv-check
  disable CHECK_VENV in catkin_virtualenv 0.6.1
* disable CHECK_VENV in catkin_virtualenv 0.6.1
* Merge pull request `#2719 <https://github.com/start-jsk/jsk_apc/issues/2719>`_ from pazeshun/install-python-tk
  [grasp_fusion, instance_occlsegm] Explicitly installs python-tk for 'import matplotlib.pyplot'
* [grasp_fusion, instance_occlsegm] Explicitly installs python-tk for 'import matplotlib.pyplot'
* Merge pull request `#2708 <https://github.com/start-jsk/jsk_apc/issues/2708>`_ from pazeshun/add-grasp_prediction_arc2017
* [grasp_fusion] Add comment of distance which camera params are optimized for
* Merge pull request `#2707 <https://github.com/start-jsk/jsk_apc/issues/2707>`_ from pazeshun/add-sphand_ros
* [grasp_fusion] Don't install data on build time to prevent build failure on travis due to googledrive access limit
  I tried travis caching of those data, but cache couldn't be expanded before travis timeout:
  https://travis-ci.org/github/start-jsk/jsk_apc/jobs/686632934
  https://travis-ci.org/github/start-jsk/jsk_apc/jobs/686826882
* Merge pull request `#2699 <https://github.com/start-jsk/jsk_apc/issues/2699>`_ from pazeshun/fix-py-venv-travis
  Fix errors in building grasp_fusion and instance_occlsegm
* Suppress pip log not to exceed maximum length in travis test
* Avoid skipping installed python requirements which is incompatible
* Add python-gdown-pip to package.xml
* Use catkin_virtualenv
* Add demos/grasp_fusion
* Contributors: Kentaro Wada, Shingo Kitagawa, Shun Hasegawa
