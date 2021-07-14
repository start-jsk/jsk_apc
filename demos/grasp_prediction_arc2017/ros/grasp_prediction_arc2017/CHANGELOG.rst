^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package grasp_prediction_arc2017
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Merge pull request `#2751 <https://github.com/start-jsk/jsk_apc/issues/2751>`_ from knorth55/fix-version
  fix version in demo packages
* fix version in demo packages
  change to 4.2.1 for all other jsk_apc packages
* Merge branch 'master' into add-sleep
* Merge pull request `#2723 <https://github.com/start-jsk/jsk_apc/issues/2723>`_ from knorth55/diable-venv-check
  disable CHECK_VENV in catkin_virtualenv 0.6.1
* disable CHECK_VENV in catkin_virtualenv 0.6.1
* Merge pull request `#2708 <https://github.com/start-jsk/jsk_apc/issues/2708>`_ from pazeshun/add-grasp_prediction_arc2017
* [grasp_prediction_arc2017] Inherit catkin_virtualenv requirements.txt of grasp_fusion
* [grasp_prediction_arc2017] Don't install data on build time to prevent build failure on travis due to googledrive access limit
* [grasp_prediction_arc2017] Remove right hand settings from book_picking.launch as they are not used
* [grasp_prediction_arc2017] N_CLASS differs between projects
* [grasp_prediction_arc2017] Fix python & euslisp installing to follow precedent and http://docs.ros.org/melodic/api/catkin/html/howto/format1/installing_python.html
* [grasp_prediction_arc2017] Use record_rosbag in sphand_driver
* [grasp_prediction_arc2017] Remove unused scale settings in book picking
* [grasp_prediction_arc2017] Add missing settings to CMakeLists.txt & package.xml
* [grasp_prediction_arc2017] Fix confliction among downloaded data
* [grasp_prediction_arc2017] hasegawa_master_thesis -> hasegawa_mthesis
* [grasp_prediction_arc2017] Add euslisp linter
* [grasp_prediction_arc2017] Add sphand_driver to run_depend
* [grasp_prediction_arc2017] Add jsons for hasegawa master thesis
* [grasp_prediction_arc2017] Fix play_rosbag_baxterlgv7
* [grasp_prediction_arc2017] Enable to use record_rosbag_baxterlgv8 in book_picking
* [grasp_prediction_arc2017] Copy record_rosbag for baxterlgv8 from https://github.com/pazeshun/jsk_apc/blob/baxterlgv8-book-picking/jsk_arc2017_baxter/launch/setup/include/record_rosbag.launch
* [grasp_prediction_arc2017] Adjust euslisp files for this package
* [grasp_prediction_arc2017] Copy euslisp files required for hasegawa master thesis from https://github.com/pazeshun/jsk_apc/tree/98aed57b34c3e390395a51f6e9b5525a47b800ee/jsk_arc2017_baxter/euslisp
* [grasp_prediction_arc2017] Add configs for hasegawa master thesis
* [grasp_prediction_arc2017] Add project input to setup_for_book_picking for selecting model
* [grasp_prediction_arc2017] Download other large data required for hasegawa master thesis
* [grasp_prediction_arc2017] Add jsons for book picking
* [grasp_prediction_arc2017] Add configs for book picking
* [grasp_prediction_arc2017] Rename setup_for_pick_baxterlgv7 to setup_for_book_picking
* [grasp_prediction_arc2017] Add main launch for book picking
* [grasp_prediction_arc2017] Adjust euslisp files for this package
* [grasp_prediction_arc2017] Copy required euslisp files from https://github.com/pazeshun/jsk_apc/tree/baxterlgv7-book-picking/jsk_arc2017_baxter/euslisp
* [grasp_prediction_arc2017] Add & use launch to bring up cameras of baxterlgv7
* [grasp_prediction_arc2017] Add baxterlgv7.launch for book picking
* [grasp_prediction_arc2017] Download other large data required for hasegawa iros2018 demo
* [grasp_prediction_arc2017] Fix install_data.py for 2nd catkin build
* [grasp_prediction_arc2017] Exclude mvtk from paths
* [grasp_prediction_arc2017] Setup for catkin build
* [grasp_prediction_arc2017] Add symbolic links to grasp_fusion
* Copy https://github.com/wkentaro/mvtk/tree/master/ros/grasp_prediction_arc2017
* Contributors: Shingo Kitagawa, Shun Hasegawa
