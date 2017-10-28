^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_2015_05_baxter_apc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.1.9 (2017-10-28)
------------------

4.1.8 (2017-10-26)
------------------

4.1.7 (2017-10-26)
------------------
* 4.1.6
* Update CHANGELOG.rst
* 4.1.5
* Update CHANGELOG.rst
* Contributors: Kentaro Wada

4.1.4 (2017-10-15)
------------------

4.1.6 (2017-10-24)
------------------

4.1.5 (2017-10-23)
------------------
* 4.1.4
* Update CHANGELOG
* Contributors: Kentaro Wada

4.1.3 (2017-10-12)
------------------

4.1.2 (2017-10-11)
------------------

4.1.1 (2017-10-10)
------------------

4.1.0 (2017-08-12)
------------------
* Revert "add timestamp functions in util.l"
  This reverts commit 203d5d2eaaa5ecd86932efb7d650644299c334fc.
* add timestamp functions in util.l
* Remove run_depend on baxter_sim_hardware to fix error on build.ros.org
  CMake Error at
  /opt/ros/indigo/share/baxter_sim_hardware/cmake/baxter_sim_hardwareConfig.cmake:141
  (message):
  Project 'jsk_2015_05_baxter_apc' tried to find library
  'baxter_sim_hardware'.  The library is neither a target nor
  built/installed
  properly.  Did you compile project 'baxter_sim_hardware'? Did you
  find_package() it before the subdirectory containing its code is
  included?
  Call Stack (most recent call first):
  CMakeLists.txt:21 (find_package)
  CMakeLists.txt:97 (select_catkin_dependencies)
* Move library to euslisp/lib for jsk_2015_05_baxter_apc
* Contributors: Kentaro Wada, Shingo Kitagawa

4.0.9 (2017-07-29)
------------------

4.0.8 (2017-07-29)
------------------

4.0.7 (2017-07-28)
------------------

4.0.6 (2017-07-28)
------------------

4.0.5 (2017-07-28)
------------------
* add argmin in util
* Contributors: Shingo Kitagawa

4.0.4 (2017-07-27)
------------------

4.0.3 (2017-07-27)
------------------

4.0.2 (2017-07-27)
------------------

4.0.1 (2017-07-26)
------------------

4.0.0 (2017-07-24)
------------------

3.3.0 (2017-07-15)
------------------

3.2.0 (2017-07-06)
------------------

3.1.0 (2017-06-30)
------------------

3.0.3 (2017-05-18)
------------------

3.0.2 (2017-05-18)
------------------
* Put in order tags in CHANGELOG.rst
* Contributors: Kentaro Wada

3.0.1 (2017-05-16)
------------------
* [jsk_2015_05_baxter_apc] remove gazebo_ros_vacuum_gripper plugin because it is already merged to upstream gazebo_plugins package.
* Contributors: Masaki Murooka

3.0.0 (2017-05-08)
------------------
* Fix rosdep key for gdown: python-gdown-pip
* Add link to wiki
* use alist to avoid segmentation fault (`#1997 <https://github.com/start-jsk/jsk_apc/issues/1997>`_)
* Contributors: Kentaro Wada, Shingo Kitagawa

2.0.0 (2016-10-22)
------------------
* Add arm2str as util and use it
* Adjust kiva pod pose to base
* Contributors: Kentaro Wada

1.5.1 (2016-07-15)
------------------
* 1.5.0
* Update CHANGELOG.rst to release 1.5.0
* 1.0.0
* Update CHANGELOG.rst
* kiva pod interactive marker
* Adjust base -> kiva_pod_base
* Adjust tf base -> kiva_pod_base
* Adjust tf base -> kiva_pod_base
* Adjust tf base -> kiva_pod_base again
* Adjust tf base -> kiva_pod_base
* 0.8.1
* update CHANGELOG
* 0.8.1
* adjust kiva pod base
* Use standalone_complexed_nodelet for Kinect2 in jsk_apc
* Put Kinect2 calib data on jsk_apc
* Adjust kinect2_torso tf and add rvizconfig for that
* Update CHANGELOG.rst for 0.8.0
* Contributors: Kei Okada, Kentaro Wada, Yusuke Niitani

1.5.0 (2016-07-09)
------------------

1.0.0 (2016-07-08)
------------------
* kiva pod interactive marker
* Adjust base -> kiva_pod_base
* Adjust tf base -> kiva_pod_base
* Adjust tf base -> kiva_pod_base
* Adjust tf base -> kiva_pod_base again
* Adjust tf base -> kiva_pod_base
* Contributors: Kentaro Wada, Yusuke Niitani

0.8.1 (2016-06-24)
------------------
* adjust kiva pod base
* Use standalone_complexed_nodelet for Kinect2 in jsk_apc
* Put Kinect2 calib data on jsk_apc
* Adjust kinect2_torso tf and add rvizconfig for that
* Update CHANGELOG.rst for 0.8.0
* Contributors: Kentaro Wada, Yusuke Niitani

0.8.0 (2016-05-31)
------------------
* kinect2_head launch use standalone complex nodelet
* kinect2_torso launch use standalone complex nodelet
* jsk_tools_add_shell_test supports from 2.0.14
* fix cmakelist depends path into full path
* 2015 launch files do not depend on 2016 config
* make .yaml compatiable with 2015 code
* Test motion for move arm to bin
* Contributors: Kei Okada, Kentaro Wada, Shingo Kitagawa, Yusuke Niitani

0.2.4 (2016-04-15)
------------------
* Add visualization tool to visualize ik to bin
* Update rosinstall
* Contributors: Kentaro Wada

0.2.3 (2016-04-11)
------------------
* Upgrade baxter SDK to 1.2.0
  * Use 1.2.0 in baxter_sim.launch
* Generate xacro robot model to generate euslisp model
  * Visualize reachable space of baxter model
  * Set predefined poses in yaml file
  * Generate eus robot model from xacro
  * Move urdf/ -> robots
  * Move urdf/ -> robots/
  * Depends when generating eus robot model from xacro
* Visualization
  * Visualize segmentation in bin
  * Set xdisplay image in 'launch/baxter.launch'
* Motion
  * Do not trust pressure sensor
  * add arm info in ros-info
  * Rename loadable-structure as .ldump -> .l
  * Add test_data for MoveArmToBin
  * Add utility functions for handling hashtable
* Refine installation
  + Fix missing depends
  + Refine rosinstall
  + Add turtlebot_description
  + Depends on roseus
  + Add missing depends
* Recognition
  + Adjust kinect2_torso
  + Adjust kiva_pod position
  + Update kiva_pod initial pos
* Documentation
  * Doc for euclid_k_clustering.py
  * Doc for initialize_baxter.py
  * Doc for work_order.py
  * Add doc for bin_contents.py
  * Add doc softlink for jsk_2015_05_baxter_apc
  * Use sphinx to make documentation
  * Checkout to a tag for demo
  * Specify version to run gazebo simulation
  * Add simulation.rosinstall
  * Set kiva:=true for 'baxter_sim.launch'
  * Add simulation.rosinstall
* Cleanup
  * Remove solidity rag merging
  * Rename json files (layout_XX.json, apc2015_layout_XX.json)
  * Remove visualize_bin_contents replaced with visualize_json
  * Remove BoF codes in this repo which is moved to jsk_perception
  * Remove README in jsk_2015_05_baxter_apc/node_scripts
* Misc
  * Install include file
  * Install files (launch,euslisp,node_scripts)

* Contributors: Kentaro Wada, Masahiro Bando

0.2.2 (2016-03-08)
------------------
* fix gmail for iory and wkentaro
* Contributors: Kei Okada

0.2.1 (2016-03-08)
------------------
* fix maintainer/author in package.xml
* Contributors: Kei Okada

0.2.0 (2016-03-08)
------------------
* Update APC 2015 for Advanced Robtoics Paper

  * Update rvizconfig for segmentation in bin
  * Update README for pick-and-verify
  * Know gripper status in control
  * Does not generate json when exists
  * Update json file with more jsons
  * Use verify-object for if grasped with point cloud
  * More jsons for pick-and-verify experiments
  * Add json files for 2016_ar
  * Fix number of trials
  * Abondont oreo
  * Update layout1
  * Update rviz config
  * Improve pick for vertical objects
  * Abondon difficult objects
  * Fix return traj speed
  * Improve picking motion
  * Large queue size
  * Use machine
  * Launch kinect2 on setup
  * Rename setup files
  * Larger queue_size
  * Swap kinect2
  * Improve return motion
  * Add limit
  * Update eps
  * Initialize tolerance
  * Stop grasp if needed
  * Remove wall picking avoidance
  * Revert avoid shelf pose
  * Euclid k cluster in main.launch
  * Fix pick-object for grasped
  * Unregister in euclid_k_clustering
  * Add catkin_INCLUDE_DIRS for std_msgs/Bool.h
  * Disable test for recognition
  * Fix roslaunch args for recognition test
  * Pass manager as argument
  * Pass manager as argument for torso
  * Update segmentation in bin gtol
  * Add layout1.json
  * Stop grasp unless grasped in bin
  * Update rviz
  * Update kinect2_head tf
  * Stable euclid k clustering
  * Detect object in bin with size feature
  * Clear params for euclid clustering
  * Stat object sizes
  * Approach to center of mass
  * EuclidKClustering with number of objects in bin
  * Use kinect2_torso for verification
  * Faster verify picked-object with pick-and-verify
  * Each view hand pose
  * In-hand object recognition with kinect2_torso
  * Input image argument for recognition_in_hand
  * Update kinect2_torso_rgb_optical_frame tf
  * More queue_size in extract_indices for bin
  * Update how to launch gazebo for APC2015

* Update for real demo on Jan 2016
  * Upgrade baxter_simulator 0.9.0 -> 0.9.1.1
  * Add gazebo vacuum gripper plugin
  * Add movie of real demo
  * Documentation how to run demo on real and sim world
  * Update demo_1.json
  * Do not verify_object unless grasping objects
  * Update real_demo.rviz
  * Remove no need tmp baxter_common version specification
  * Add README for jsk_2015_05_baxter_apc
  * Use jsk_recognition_msgs/ClassificationResult for color_hist
  * Fix wait-for-opposite-arm
  * Add sample of picking with clustering points
  * Update color_histogram object recognition for multi regions
  * Use boost_object_recognition in object_recognition
  * Update boost object recognition as transport
  * Fix color_object_matcher as transport
  * Boost object recognition
  * [jsk_2015_05_baxter_apc] Add place-object method
    Modified:
    - jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l
  * Launch visualize_json.py
  * Add queue_size option for recognitions
  * Update demo_1.json
  * Add option of queue_size
  * Update demo-1 json
  * Add INPUT_DEPTH arg for torso kinect2
  * Update tf of kinect2 torso
  * Fix opencl error on kinect2 head
  * Rename function name object_list -> get_object_list
  * Add demo_1.json
    Added:
    - jsk_2015_05_baxter_apc/json/demo_1.json
  * Respawn object recognition nodes
    Modified:
    - jsk_2015_05_baxter_apc/launch/include/object_recognition.launch
  * Longer spin off for object grasped
    Modified:
    - jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l
    - jsk_2015_05_baxter_apc/euslisp/main.l
  * Add picking method with solidity rag merging and its example
  * Launch solidity_rag_merge for grasp planning with vacuum gripper
  * Update kinect2_head position on 2016-01-27
  * Update self filter padding
  * Remove kiva_pod joint_states
  * Update kinect2_torso tf
  * Add in_bin_vision.launch
  * Update ik to bin
  * Faster verify pose
  * [jsk_2015_05_baxter_apc] Do not depends on mahotas
  * [jsk_2015_05_baxter_apc] Extract the cached test_data
  * [jsk_2015_05_baxter_apc] Fix broken topic names
  * [jsk_2015_05_baxter_apc] Test time-limit 60 -> 360
  * [jsk_2015_05_baxter_apc] Add jsk_tools as test_depend
  * [jsk_2015_05_baxter_apc] Use cached test_data
  * [jsk_2015_05_baxter_apc] Use bof_object_matcher in jsk_perception
  * [jsk_2015_05_baxter_apc] Real demo rviz config
  * Add retry 3 for recognition test by BOF
  * Update gazebo_demo.rviz
  * Add fold/reset/untuck pose script
  * Add FIXME
  * Minor change of apc_gazebo world
  * Update rviz config for gazebo demo
  * Fix typo
  * Add rviz config for gazebo
  * Add visualization script on rviz
  * Put objects in all bins
  * [jsk_2015_05_baxter_apc] Add order-bin and stage to the world
  * [jsk_2015_05_baxter_apc] Add paper mate
  * Remove no need static
  * [jsk_2015_05_baxter_apc] Fixed end effector and baxter base
  * Fix eus for gazebo
  * [jsk_2015_05_baxter_apc] Move interactive_marker config
  * [jsk_2015_05_baxter_apc] Fix transform world to base invalid arg
  * [jsk_2015_05_baxter_apc] Set camera_name
  * Adjust kinect
  * [jsk_2015_05_baxter_apc] Put kiva correct place and safety glass also
  * [jsk_2015_05_baxter_apc] Fix typo
  * Add left state publisher
  * Set /apc_on_gazebo param
  * [jsk_2015_05_baxter_apc] Rename to baxter_sim.launch
  * [jsk_2015_05_baxter_apc] Add gazebo mode vacuum gripper
  * Update test_data
  * [jsk_2015_05_baxter_apc] Refactr urdf files
  * [jsk_2015_05_baxter_apc] Add fold-pose-back.l
  * [jsk_2015_05_baxter_apc] Add right_end_effector and vacuum_gripper
  * Recognize bins at first
  * Adjust kiva pos
  * Enhance picking
  * Fix bbox x, z comparison
  * Recognize bins at first
  * Adjust kiva pos
  * Enhance picking
  * Fix bbox x, z comparison
  * [jsk_2015_05_baxter_apc] Pass timestamp to recognition method
  * [jsk_2015_05_baxter_apc] Adjust place-object-pose
  * [jsk_2015_05_baxter_apc] Adjust place-object-pose
  * Use robot_self_filter package
  * [jsk_2015_05_baxter_apc] Remove approximate_sync (no need)
    This is no need with change in
    PR2/pr2_navigation/pr2_navigation_self_filter
    Related to https://github.com/PR2/pr2_navigation/pull/24
* Recognition in bin for APC2015
  * [jsk_2015_05_baxter_apc] Run main as script
  * [jsk_2015_05_baxter_apc] Add script to move arm and do verify pose
  * Add timeout
  * Add mahotas as run_depend
  * Remove duplicate rostest declaration
  * Add gdown as run_depend
  * Run depends on imagesift
  * [jsk_2015_05_baxter_apc] Run test actually
  * [jsk_2015_05_baxter_apc] Make color_object_matcher as transport
  * [jsk_2015_05_baxter_apc] Test recognitioin in hand
  * Rename scripts -> node_scripts
  * [jsk_2015_05_baxter_apc] Update kinect2_torso tf
  * [jsk_2015_05_baxter_apc] fix approach to object
  * [jsk_2015_05_baxter_apc] Fix return object avoid shelf
  * [jsk_2015_05_baxter_apc] Fix typo
  * [jsk_2015_05_baxter_apc] Custom baxter urdf for gazebo world
  * jsk_2015_apc_common -> jsk_apc2015_common
  * Add catkin_lint
  * [jsk_2015_05_baxter_apc] Fix return height
  * [jsk_2015_05_baxter_apc] Work :try-to-pick
  * [jsk_2015_05_baxter_apc] Go to wait after all orders
  * [jsk_2015_05_baxter_apc] Add doura.launch
  * [jsk_2015_05_baxter_apc] Update segmentation_in_bin.rviz
  * [jsk_2015_05_baxter_apc] Remove self filter from baxter.launch
  * [jsk_2015_05_baxter_apc] Make faster localization in hand
    * use self_filter in bottom
  * [jsk_2015_05_baxter_apc] Specify max_depth in kinect2_bridge.launch
    Remove points_reachable
  * Revert "[jsk_2015_05_baxter_apc] filter by x"
    This reverts commit 590ad8d96b56a72ba47eb5bd1864b51657ff56df.
  * [jsk_2015_05_baxter_apc] Visualize objects and bins
  * [jsk_2015_05_baxter_apc] Fix :get-next-order
  * [jsk_2015_05_baxter_apc] filter by x
  * [jsk_2015_05_baxter_apc] Split segmentation in bin for atof and gtol
  * [jsk_2015_05_baxter_apc] Add kiva_pod_state.launch
  * [jsk_2015_05_baxter_apc] See same package config dir
  * [jsk_2015_05_baxter_apc] Add rvizconfig to adjust kiva pod
  * [jsk_2015_05_baxter_apc] Update box position for g to l
  * [jsk_2015_05_baxter_apc] Segmentation for A to F
  * [jsk_2015_05_baxter_apc] 1.2 passthrough z
  * [jsk_2015_05_baxter_apc] Use self_filtered points
  * [jsk_2015_05_baxter_apc] min_size 200 -> 500
  * [jsk_2015_05_baxter_apc] Initialize param in main.launch
  * [jsk_2015_05_baxter_apc] Stop using kiva_pod_filter
  * [jsk_2015_05_baxter_apc] Fix verify-object
  * [jsk_2015_05_baxter_apc] Remove timeout in recognize-object-in-hand
  * [jsk_2015_05_baxter_apc] pick wall near object
  * [jsk_2015_05_baxter_apc] stop-grasp to place
  * [jsk_2015_05_baxter_apc] middle is right work
  * [jsk_2015_05_baxter_apc] left_process -> left_hand
  * [jsk_2015_05_baxter_apc] typo
  * [jsk_2015_05_baxter_apc] typo
  * [jsk_2015_05_baxter_apc] typo
  * [jsk_2015_05_baxter_apc] Fix typo
  * [jsk_2015_05_baxter_apc] namespace change
  * [jsk_2015_05_baxter_apc] Add :try-to-pick-in-bin
  * [jsk_2015_05_baxter_apc] Add :try-to-pick-object
  * [jsk_2015_05_baxter_apc] Archive test file
  * [jsk_2015_05_baxter_apc] Archive test file
  * [jsk_2015_05_baxter_apc] Archive test file
  * [jsk_2015_05_baxter_apc] Fix main params
  * [jsk_2015_05_baxter_apc] z direction pick object
  * [jsk_2015_05_baxter_apc] Stop using one-shot-publish
  * [jsk_2015_05_baxter_apc] Fix include path
  * [jsk_2015_05_baxter_apc] Fix tf-transform
  * [jsk_2015_05_baxter_apc] :recognize-object-in-bin topic change
  * [jsk_2015_05_baxter_apc] :recognize-bin-boxes topic change
  * [jsk_2015_05_baxter_apc] Update setup.launch for latest software
  * [jsk_2015_05_baxter_apc] Refactor baxter.launch
  * [jsk_2015_05_baxter_apc] Add segmentation_in_bin.launch
  * [jsk_2015_05_baxter_apc] Remove object_segmentation.launch
  * [jsk_2015_05_baxter_apc] Add segmentation_in_hand.launch
  * [jsk_2015_05_baxter_apc] Move deprecated launch files
  * [jsk_2015_05_baxter_apc] Move meshes location
  * [jsk_2015_05_baxter_apc] Remove upload_baxter.launch
  * [jsk_2015_05_baxter_apc] Launch vacuum_gripper in baxter.launch
  * [jsk_2015_05_baxter_apc] Rename to vacuum_gripper.launch
  * [jsk_2015_05_baxter_apc] Add self_filter.launch
  * [jsk_2015_05_baxter_apc] Filter reachable clouds
  * [jsk_2015_05_baxter_apc] Remove base_footprint
  * [jsk_2015_05_baxter_apc] Add jsk_rqt_plugins to run_depend
  * [jsk_2015_05_baxter_apc] Archive motion codes
  * [jsk_2015_05_baxter_apc] Archive setup_params.py
  * [jsk_2015_05_baxter_apc] Refactor mainloop
  * [jsk_2015_05_baxter_apc] Remove speak-en
  * [jsk_2015_05_baxter_apc] Use one-shot-subscribe to get bin_contents
  * [jsk_2015_05_baxter_apc] Use one-shot-subscribe in :get-work-orders
  * [jsk_2015_05_baxter_apc] Use one-shot-subscribe in recognize-objects-in-bin
  * [jsk_2015_05_baxter_apc] arm-symbol-to-str -> arm-symbol2str
  * [jsk_2015_05_baxter_apc] Use one-shot-publish to control gripper
  * [jsk_2015_05_baxter_apc] Add _ prefix for slots
  * [jsk_2015_05_baxter_apc] Use one-shot-subscribe for recognize-bin-boxes
  * [jsk_2015_05_baxter_apc] Add get-a-work-order
  * [jsk_2015_05_baxter_apc] Add :wait-for-user-input-to-start
  * [jsk_2015_05_baxter_apc] symbol2str, str2symbol
  * [jsk_2015_05_baxter_apc] Add :get-target-bin
  * [jsk_2015_05_baxter_apc] kinect2 -> kinect2_head
  * [jsk_2015_05_baxter_apc] Add concatenate_clouds.launch
  * [jsk_2015_05_baxter_apc] Remove kinect2_tf.launch
  * [jsk_2015_05_baxter_apc] Archive robot-recognition.l
  * [jsk_2015_05_baxter_apc] Methodize real-sim-end-coords-diff
  * [jsk_2015_05_baxter_apc] Rename robot-main.l -> main.l
  * [jsk_2015_05_baxter_apc] Methodize graspingp
  * [jsk_2015_05_baxter_apc] Methodize verify-object
  * [jsk_2015_05_baxter_apc] Remove robot-init.l
  * [jsk_2015_05_baxter_apc] Remove utils.l and robot-utils.l
  * [jsk_2015_05_baxter_apc] Adjust kinect2_head tf
  * Add object_segmentation.launch
  * Update kinect2 torso tf
  * Use cpu for kinect2 torso
  * [jsk_2015_05_baxter_apc] Add roslaunch for kinect2_head
  * arg default -> value
  * [jsk_2015_05_baxter_apc] Add iai_kinect2 in rosinstall
  * [jsk_2015_05_baxter_apc] roslaunch for kinect2_torso
    Closes `#907 <https://github.com/start-jsk/jsk_apc/issues/907>`_
    Closes `#909 <https://github.com/start-jsk/jsk_apc/issues/909>`_
  * [jsk_2015_05_baxter_apc] Error catch when object cloud is not found
  * [jsk_2015_05_baxter_apc] Fix test for new *ri* :pick-object
  * [jsk_2015_05_baxter_apc] Add pick-object method
  * Flexible env var for APC shelf model for Gazebo
  * Pick object from object :z axis
  * Improve ik for bin entrance
  * [jsk_2015_05_baxter_apc] Remove robot-input
  * Add :avoid-shelf-pose to avoid shelf collision
  * Add :arm-symbol-to-str
  * (:ik-avs->object-in-bin) to pick object
  * Recognize bin boxes once and memorize these position
  * Refactor: Remove baxter :locate from robot-init
  * bin-entrance is half of dim-x distance from the center
  * [jsk_2015_05_baxter_apc] Remove update-score
  * [jsk_2015_05_baxter_apc] Remove robot-communication.l
  * [jsk_2015_05_baxter_apc] Remove (return-object)
  * Refactor: Remove orderbin
  * Refactor: Remove visualization lines
  * Refactor: Remove *tfb*
  * (move-for-verification) -> (send *ri* :move-arm-body->head-view-point)
  * [jsk_2015_05_baxter_apc] remove (look-at-other-side)
  * [jsk_2015_05_baxter_apc] remove (look-at-other-side)
  * [jsk_2015_05_baxter_apc] Remove (rotate-wrist)
  * (place-object) -> (send *ri* :move-arm-body->order-bin)
  * (send *ri* :move-to-bin) -> (send *ri* :move-arm-body->bin)
  * [jsk_2015_05_baxter_apc] Use :hard-coded-pose method
  * [jsk_2015_05_baxter_apc] Use :l/r-reverse
  * [jsk_2015_05_baxter_apc] Add .gitignore to test dir
  * [jsk_2015_05_baxter_apc] Add TODO for baxter location
  * [jsk_2015_05_baxter_apc] Download rosbag and make the test passes
  * [jsk_2015_05_baxter_apc] Remove :untuck-pose
  * [jsk_2015_05_baxter_apc] Fix bin-box using copy-object
  * [jsk_2015_05_baxter_apc] Remove move-to-target-bin function
  * [jsk_2015_05_baxter_apc] Remove position decision tool
  * [jsk_2015_05_baxter_apc] Complete :move-to-bin method
  * [jsk_2015_05_baxter_apc] Remove untuck-pose
  * [jsk_2015_05_baxter_apc] Remove fold-to-keep-object-av
  * [jsk_2015_05_baxter_apc] (load "..") -> (require "..")
  * [jsk_2015_05_baxter_apc] Refactor: (apc-init)
  * [jsk_2015_05_baxter_apc] Refactor: remove (fold-pose-back)
  * [jsk_2015_05_baxter_apc] Remove fold-pose-* functions
  * [jsk_2015_05_baxter_apc] Add :fold-pose-* methods
  * [jsk_2015_05_baxter_apc] fix path and name changed class
  * [jsk_2015_05_baxter_apc] Add subclasses
  * [jsk_2015_05_baxter_apc] robot-interface.l -> baxter-interface.l
  * [jsk_2015_05_baxter_apc] Add baxter-interface.l
  * [jsk_2015_05_baxter_apc] move model
  * [jsk_2015_05_baxter_apc] Move rosinstall to package dir
  * [jsk_2015_05_baxter_apc] run_depend jsk_pcl_ros
  * [jsk_2015_05_baxter_apc] Use jsk_2015_apc_common.data:object_list
  * Move mesh files jsk_2015_05_baxter_apc -> jsk_2015_apc_common
  * Adjust kinect2 tf and baxter custom link after calibration of kinect2
  * Publish tf's at launch of baxter.launch
  * Rename pkg: jsk_2014_picking_challenge -> jsk_2015_05_baxter_apc
* Contributors: Isaac IY Saito, Kentaro Wada

0.1.1 (2015-09-14)
------------------
* Remove actionlib msgs which is not used
* Sort depends in alphabetical order
* Show debug info for object recognition
* Change weight of rolodex_jumbo_pencil_cup
* Remove no need dependencies and add jsk_recognition_msgs
* [euslisp/robot-init.l] Baxter position in lab
* Add toggle_vacuum.py
* Fix test-robot-motion
* Fix jsk_rqt_plugins.srv YesNo
* Contributors: Kentaro Wada

0.1.0 (2015-06-11)
------------------
* [CMakeLists.txt] Add roseus in find_package
* [data/apc.json] Add real challenge json file
* final change
* return-object change depth
* fix cons bug
* fix target-bounding-box
* fix baxter height to 1030
* fix wrong setup.launch
* final check of pick-object
* add stop-grasp for test
* Fix error in bbox
* Tuning paramter of bounding box in doura
* modified pick-object's faint movement
* modified pick-object doesn't work because bounding-box-hint is nil
* [launch/main.launch] json arg is required
* [scripts/check_shelf_pos.l] fix to work with baxter with differnt height using ik
* [scripts/test_object_recognition.py] Remove duplicate script
* [robot-init.l] Adjust baxter & pod pos for the real challenge
* fix pick-offset error caused by check-if-grabed's arguments change
* add check-pick-object-offset-from-wall to adjust parameters
* add bounding box hint callback
* Contributors: Kei Okada, Kentaro Wada, Yuto Inagaki, Iori Yanokura

0.0.2 (2015-05-24)
------------------
* 2015--5-24 16:07 working version
* Contributors: Kei Okada, Kentaro Wada, Noriaki Takasugi, Yuto Inagaki, Iori Yanokura, Jiang Jun

