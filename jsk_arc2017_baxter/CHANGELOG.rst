^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_arc2017_baxter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3.1.0 (2017-06-30)
------------------
* Fix for euslint
* Update data collection motion
* Change save_dir in dynamic
* Update motion
* Use last 3 frames as texture
* Generate texture model of objects by kinfu
* move set segmentation candidates method
* update UpdateJSON and replace SaveJSON by Trigger
* correct indent in stow-interface.l
* use fcn in stow task recognition pipeline
* remove unused parameters in setup_for_stow
* move hand camera nodes to setup launch
* update stow_task environment config
* add json_saver methods and save json in main loop
* add json_saver.py
* use latest fcn model for segmentation
* change state-machine frequency: 1.0 -> 2.0 hz
* add path-constraints for place object
* update pick motion parameters for new env
* update cardboard moveit methods
* update cardboard pos for new env
* update shelf_bin and shelf_marker for new shelf
* fix typo in baxter.launch
* Merge pull request `#2154 <https://github.com/start-jsk/jsk_apc/issues/2154>`_ from wkentaro/test_task_arc_interface
  Add test for motion code in both pick and stow tasks
* add baxter-moveit-environment for gripper-v6
* update right_vacuum_gripper.xacro for gazebo
* add baxter_sim.launch in jsk_arc2017_baxter
* add moveit config for gripper-v6
* Remove no need newline in tote.yaml
* Merge branch 'master' into test_task_arc_interface
* Don't load old robot model
* Revert mvit-env and mvit-rb
* Adjust gravity compensation automatically
* Fix parenthesis and add comment to move-hand
* Adjust rvizconfig to gripper-v6
* Fix arc-interface to support left hand
* Use only left astra mini
* Apply IK additional check to avoid collision to bin wall
* Use wait-interpolation-until-grasp to prevent unnecessary push
* Fix wait-interpolation-until-grasp for first interpolatingp nil
* Fix rarm pressure threshold
* Use right_hand_left_camera in setup_for_stow
* Fold fingers in picking to avoid collision
* Add finger init motion to pick and stow init
* Use right_hand_left_camera in setup_for_pick
* Disable rviz in default of stereo_astra_hand
* Fix linter target
* Adjust euslisp codes to baxter with right gripper-v6
* Add baxter.launch for right gripper-v6
* Add ros_control layer for gripper-v6
* Add dxl controller for gripper-v6
* Add baxter model with right gripper-v6
* Place location config files in jsk_arc2017_baxter
* state_server accept Ctrl-C keyboard interruption
* remove duplicated line
* update stow-arc-interface test
* add publish_tote_boxes and interactive tote marker
* Add test for arc-interface for stow task
* Generalize visualize-bins by renaming it to visualize-boxes
* Publish source location of task in setup_for\_(pick|stow).launch
* Fix typo and test arc_interface for pick task
* Move task config to jsk_arc2017_baxter
* Yet another refactoring of stereo_astra_hand.launch
* add "task" argument to select shelf_marker.yaml
* Refactoring right_hand rgb-d camera stereo
* fix typo
* add files for data collection
* Update tf from right to left by using project matrix
* Update transformation from left_hand to right_hand
* Use moveit to avoid collision to box and shelf
* Collect data in shelf bins
* Fix typo in filename
* Update rvizconfig name
* Update rvizconfig
* Reuse possible code by using include in roslaunch file
* Don't use laser
* Refactor stereo_astra_hand.launch
* Remove spam.launch
* Improve visualization of triple fusion
* support quad fusion
* update calibration yaml files
* Quad fusion using depth from laser scan
* test for laser depth_image_creator
* add tilt laser to stereo system
* Launch right stereo camera in baxter.launch
* calibrated extrinsic parameter
* add depth image merging nodes
* add monoral_camera_info files
* move stereo_camera_info files from jsk_2016_01_baxter_apc to jsk_arc2017_baxter
* move stereo_astra.launch to launch/setup/ directory
* introduce stereo astra_mini_s
* Add create_udev_rules and simplify README
* Merge pull request `#2152 <https://github.com/start-jsk/jsk_apc/issues/2152>`_ from pazeshun/fix-bugs-stow
  Fix small bugs added when adding stow
* Don't change target-obj in verify-object
* Revert offsets for bin overlook pose
* Fix mistakes of arg and return value
* Use fold-pose-back in arc-interface
* Fix translation in ik->bin-center and ik->tote-center
* add moveit-p slot in stow-interface
* add moveit-p slot in pick-interface
* Add Arduino sketch for sparkfun sensor
* Remove unused constants and functions in firm
* Lighten GripperSensorStates msg
* add main program state machine test
* add state_server test for stow task
* fix indent of main launch files
* use symbol-string to replace string-upcase
* translate bin/tote coords in local coordinate
* fix typo in arc-interface
* add stow.launch and stow.rviz
* add stow-main.l
* add stow-interface.l
* update pick methods and add :pick-object-in-tote
* add stow_task methods and slots
* mv ik->cardboard-entrance -> ik->cardboard-center
* replace :ik->bin-entrance by  ik->bin-center
* use bin-cubes- instead of bin-boxes-
* reset order in wait-for-user-input
* rename to :recognize-target-object and update
  :recognize-objects-in-bin -> :recognize-target-object
* update pick-main state machine
* state_server support stow_task and set rosparam
* add shelf_marker for stow_task
* fail-count -> picking-fail-count for pick task
* add setup_for_stow launch
* add &rest args in :fold-pose-back method
* move fold-pose-back method in arc-interface
* Publish proximity sensor values with other gripper sensor states (`#2125 <https://github.com/start-jsk/jsk_apc/issues/2125>`_)
  * add FA-I sensor to gripper-v5
  * add GripperSensorStates republish program
  * Rename and refactor republish_gripper_sensor_states.py
  * rename finger flex topic
  * add eof to .travis.rosinstall
* fix typo in pick-interface.l (`#2133 <https://github.com/start-jsk/jsk_apc/issues/2133>`_)
* add roseus_smach run_depend in package.xml
* add lint test for node_scripts
* add state_server test
* add :get-state method in arc-interface
* add FIXME smach_viewer in main.launch
* add state_server in main.launch
* use smach state-machine in pick-main.l
* add state_server methods in arc-interface
* add state_server.py
  this server collect state of both arms
  and determine which arm can start picking
* add UpdateState GetState and CheckCanStart srv
* add pick-interface
* move :send-av in arc-interface
* use baxter-robot for init robot and add FIXME
* add :spin-off-by-wrist in arc-interface
* arc-interface inherits propertied-object
* use *ri* *baxter* in arc-interface
  I follwed *tfl* usage in robot-interface.l.
* use global var *tfl* set in robot-interface
* rename *arc* -> *ti*
  *ti* is named after task-interface
* use robot of slots in baxter-interface
* split arc-interface and baxter-interface
* Add Arduino firmware for right gripper-v6
* fix bug in pick-main
* update move overlook method to support all bins
* modify :ik->bin-entrance
* do not wait head motion
* modify movable region
* modify overlook-pose
* move point-shelf-position.l
* rename detect-bin-position -> point-shelf-position
* add require lines and show warn message
* redefine detect-bin-position() in another file
* point ideal position of bin
* set movable region for bin narrower in order not to collide with bin
* improve motion in :place_object
* remove inefficient motion in :recognize_objects_in_bin
* calibration for rarm in the beginnig, and after that larm. not simultaneously.
* use key in pick-init
* use angle-vector-raw in pick method
* fix typo in moveit methods
* add pick.rviz in jsk_arc2017_baxter
* set default arg moveit as true
* add moveit arg in pick launch
* add moveit scenes in pick-main
* add moveit methods in arc-interface
* rename detect-bin-position -> point-shelf-position
* add require lines and show warn message
* redefine detect-bin-position() in another file
* point ideal position of bin
* do not wait head motion
* modify movable region
* modify overlook-pose
* use key in pick-init
* use angle-vector-raw in pick method
* fix typo in moveit methods
* add pick.rviz in jsk_arc2017_baxter
* set default arg moveit as true
* add moveit arg in pick launch
* add moveit scenes in pick-main
* add moveit methods in arc-interface
* refine place_object motion (`#2103 <https://github.com/start-jsk/jsk_apc/issues/2103>`_)
  * remove and move rosparam and add TODO in pick-main
  * refine place_object motion
* fix :pick_object (`#2101 <https://github.com/start-jsk/jsk_apc/issues/2101>`_)
* Contributors: Kei Okada, Kentaro Wada, Naoya Yamaguchi, Shingo Kitagawa, Shun Hasegawa, Yuto Uchimi, YutoUchimi

3.0.3 (2017-05-18)
------------------
* Add roseus as build_depend
* update midpose to go back fold-pose-back (`#2093 <https://github.com/start-jsk/jsk_apc/issues/2093>`_)
* Contributors: Kentaro Wada, Shingo Kitagawa

3.0.2 (2017-05-18)
------------------

3.0.1 (2017-05-16)
------------------
* Move astra_hand.launch from setup_for_pick.launch to baxter.launch
* fix typo in CMakeLists
* Fix for moved euslint to jsk_apc2016_common
* Depends at test time on jsk_2016_01_baxter_apc
* add wait condition for wait_for_user_input
* got to wait_for_opposite_arm first
* update waiting condition
* fix typo in arc-interface
* mv euslint to jsk_apc2016_common package
* Contributors: Kentaro Wada, Shingo Kitagawa, YutoUchimi

3.0.0 (2017-05-08)
------------------
* add TODO in util.l
* rename opposite-arm -> get-opposite-arm
* move get-bin-contents to arc-interface
* format apc -> arc for ARC2017
* remove unused package and sort alphabetically
* add find_package jsk_2016_01_baxter_apc in test
* refer related issue in TODO
* move some util func in apc-interface
* add TODO: make apc-inteface and pick-interface class properly
* make tf->pose-coords as a method of apc-interface
* rename arg launch_main -> main
* set myself as a author
* mv pick_work_order_server -> work_order_publisher
* replace publish_shelf_bin_bbox to existing node
* improve euslint to accept path
* remove unnecessary lines in CMakeLists
* update pytorch fcn model file
* place manager in ns
* fix and improve let variables
* use arm2str instead of arm-symbol2str
* improve picking motion
* when object is not recognized, wait opposite arm
* rename get-movable-region -> set-movable-region
* modify pick object motion
* angle-vector use :fast and :scale
* update overlook-pose to avoid aggresive motion
* rename baxter-interface -> apc-interface
* fix typo and improve euslisp codes
* fix typo in pick.launch for jsk_arc2017_baxter
* add pick.launch for arc2017
* add euslint in jsk_arc2017_baxter
* add euslisp codes for arc2017
* add myself as a maintainer
* update CMakelists.txt and package.xml for roseus
* move baxter.launch to setup
* add setup_for_pick.launch for arc2017
* add baxter.launch for arc2017
* move collect_data_in_bin in launch/main
* add run_depend in jsk_arc2017_baxter
* Add link to wiki
* Fix typo in collect_data_in_bin.launch
* Save tf and bin_name also
* Save tf also
* Save data with compression
* Update save dir
* Add data_collection program in bin
* Contributors: Kentaro Wada, Shingo Kitagawa
