^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package sphand_driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Merge pull request `#2751 <https://github.com/start-jsk/jsk_apc/issues/2751>`_ from knorth55/fix-version
  fix version in demo packages
* Merge branch 'master' into fix-version
* Merge pull request `#2722 <https://github.com/start-jsk/jsk_apc/issues/2722>`_ from knorth55/add-sleep
  add sleep in calib_required_controller
* add sleep in calib_required_joint_controller in sphand_driver
* fix version in demo packages
  change to 4.2.1 for all other jsk_apc packages
* Merge pull request `#2708 <https://github.com/start-jsk/jsk_apc/issues/2708>`_ from pazeshun/add-grasp_prediction_arc2017
* Change topic name 'from_main' -> 'gripper_front'
* Merge pull request `#2707 <https://github.com/start-jsk/jsk_apc/issues/2707>`_ from pazeshun/add-sphand_ros
* [sphand_ros] Add reflectance_param to format_printed_prox_to_csv
* [sphand_ros] Speed up dynamixel loop to 40Hz
* [sphand_ros] prop_const -> reflectance_param
* [sphand_ros] Add euslisp interface to get prop_const
* [sphand_ros] Record compressedDepth as it is now not relayed from UP Board
* [sphand_ros] Don't use compressedDepth from UP Board
  When subscribing compressedDepth from kinetic UP Board, hz of image_raw drops.
  This is because usage of CPU processing camera becomes 100%.
  Even on baxter-c1, hz of image_raw drops if png_level is 9 or 8.
* [sphand_ros] Add euslisp interface of combined distance
* [sphand_ros] Fix to use ToF in combined distance when no obj is found after calibration
* [sphand_ros] Publish distance combined with ToF output
  Far part: ToF distance, middle & close part: distance generated from intensity
* [sphand_ros] ws_jsk_apc -> kinetic
* [sphand_ros] Fix installing to follow http://docs.ros.org/melodic/api/catkin/html/howto/format1/index.html
* [sphand_ros] Add launch recording rosbag
* [sphand_ros] Add install settings
* [sphand_ros] Add test_depend
* [sphand_ros] Remove dependency for msg generation from sphand_driver
* [sphand_ros] Remove dependency to mraa from package.xml
  Because adding apt repository is required for mraa and rosdep cannot resolve dependency
* [sphand_ros] Do TODOs of package.xml
* [sphand_ros] Enable linter to check all python files
* [sphand_ros] Add tests
* [sphand_ros] Fix proximity printer usage
* [sphand_ros] Copy proximity printer from https://github.com/pazeshun/jsk_apc/tree/baxterlgv8-book-picking/jsk_arc2017_baxter
* [sphand_ros] Adjust euslisp files
* [sphand_ros] Copy required euslisp files from https://github.com/pazeshun/jsk_apc/tree/baxterlgv8-book-picking/jsk_arc2017_baxter/euslisp
* [sphand_ros] Add launch of task-agnostic part of baxterlgv7.launch
* [sphand_ros] Add launch to create point cloud on remote PC
* [sphand_ros] Follow workspace path change
* Copy sphand_ros from https://github.com/pazeshun/sphand_ros
* Contributors: Shingo Kitagawa, Shun Hasegawa
