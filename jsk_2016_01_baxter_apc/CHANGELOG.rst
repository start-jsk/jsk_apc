^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_2016_01_baxter_apc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.8.1 (2016-06-24)
------------------
* add roslint to package.xml
* update maintainers
* changed pressure threshold 840 -> 810
* Use wait-interpolation-smooth for objects not to run away from gripper
* Check the grasp before saving data
* Use stamped filename for video recording with axis camera
* Save hand pose at each view hand pose
* Change save directory at each time of picking
* Add script to randomly change the view hand pose
* Merge pull request `#1775 <https://github.com/start-jsk/jsk_apc/issues/1775>`_ from wkentaro/fix-grasp-log
  Fix writing grasp success/fail log when file does not exist
* Merge pull request `#1773 <https://github.com/start-jsk/jsk_apc/issues/1773>`_ from wkentaro/remove-fold-pose
  Remove fold-to-keep pose at each time for viewing
* Fix writing grasp success/fail log when file does not exist
* Remove fold-to-keep pose at each time for viewing
* update path for trained segmenter pkl
* Use :to-nec to strify the rostime in roseus
* Enable to get floating bounding box
* Add cube->cube-parallel-to-coords method
* Adjust depth frame of astra cameras on 2016-06-22 00:17:11
* right left hand rgb/depth calib
* changed vacuum_gripper.launch not to launch rosserial_node3
* Add :visualize-path method to jsk_2016_01_baxter_apc::baxter-robot
* collect sib data launch
* added firmware of arduino which controls vacuum switch
* Don't rotate objects in Bin
* Trust pressure sensor again
* Enable to use kinect in picking-with-sib.l
* get graspingp after second approach
* Write grasp success/fail log while data collection on table
* sib kinect
* Merge pull request `#1750 <https://github.com/start-jsk/jsk_apc/issues/1750>`_ from wkentaro/stop-grasp-in-data-collection
  Stop grasp unless grasped object when picking
* Stop grasp unless grasped object when picking
* Retry when ik failed to place object on table
* Look for view pose to detect table center
* Control vacuum gripper with a script
* removed image resizer from launch because astra does not need them
* Fix motion to Bin k
* Fix motion to Bin e
* add local variable in :need-to-wait-opposite-arm
* Fix typo
* Enhance naming of method :place-object-on-plane -> :place-object-on-table
* Fix typo
* Enhance the order of sleep and gripper servo on
* Add data collection program for in-hand object recognition
* Add reset-pose script
* Stop doing self_filter while recognizing object in hand
* Merge pull request `#1727 <https://github.com/start-jsk/jsk_apc/issues/1727>`_ from wkentaro/respawn-astra-2
  Respawn true for astra camera
* Respawn true for astra camera
* Fix typo in astra_hand.launch
* Launch vgg16_object_recognition in satan
* color frame fixed
* add setup_astra launch file
* Push gripper joint states back of other joint states
* Adjust depth_frame of hand cameras
* rename set_bin_param -> publish_bin_info
* publish bin bbox node split from publish bin info
* add astra check launch and rvizconfig
* use astra camera instead of softkinetic
* changed threshold of pressure
* Detect grasps with pressure threshold 840 [hPa]
  For `#1699 <https://github.com/start-jsk/jsk_apc/issues/1699>`_
* Adjust baxter-interface to SPB2f
* Adjust end-coords to SPB2f
* Change collision link of vacuum pad to SPB2f
* use publish_bin_info node for :recognize-bin-boxes
* add pick task json output node
* Avoid collision to Bin top
* Collect hard-coded variables to slot
* Add left gripper to gripper jta server
* Add left gripper to gripper_joint_states_publisher.cpp
* Add left gripper to enable_gripper.cpp
* added offset for left gripper servo
* added firmware of left gripper-v3 arduino
* Add new arduino node to baxter.launch
* Adjust left arm motion to right
* Adjust angle-vector in test-again-approach to new robot
* Rename test-ik -> test-again-approach-bin-l
* Rotate left gripper servo in test-ik-in-bin
* Fix :arm-potentio-vector to get proper vector
* Fix :rotate-wrist not to depend on joint num
* Add lgripper-controller to baxter-interface
* Add left gripper joint to baxter.yaml and adjust left arm pose to right
* Add gripper-v3 to left arm
* Add gripper-v3 meshes
* Add left gripper to in_hand_clipper
* Add left gripper to self filter
* Merge pull request `#1644 <https://github.com/start-jsk/jsk_apc/issues/1644>`_ from knorth55/servo-separate
  split gripper-servo-off and gripper-servo-on from certain method
* use local variable in :recognize-objects-segmentation-in-bin
* split gripper-servo-on from :spin-off-by-wrist
* Merge pull request `#1633 <https://github.com/start-jsk/jsk_apc/issues/1633>`_ from pazeshun/use-clustering
  Enable to use clustering instead of SIB
* split gripper-servo-off from :move-arm-body->order-bin
* Output simple error message if unable to get param
* Disable test_move_arm_to_bin
* Add setup_head.launch to jsk_2016_01_baxter_apc
* Adjust baxter-interface to new bin model
* add option :use-gripper in :inverse-kinematics
* Enable to use clustering instead of SIB
* fixed sib_softkinetic_test to not publish errors
* Fill time_from_start in feedback
* Sleep until trajectory start time
* Publish feedbacks continuously among command points
* Fix extendability of gripper_trajectory_server.cpp
* Fix indent of gripper_trajectory_server.cpp
* modify gripper-angle to 90 in overlook pose
* fix style in euslisp/*.l and test/*.l
* euslint test only euslisp/*.l and test/*.l
* add white space, line length and indent test in euslint and improve result output
  indent test is diabled
* euslint style fix
* stop-grasp only one arm in return_object
* update main.launch to call layout visualizer in 2016
* add timeout in method :recognize-objects-segmentation-in-bin
* edit download_test_data.py
* test for sib_softkinetic
* make sib_visualization modular & fix indent
* visualize posterior overlaid with color
* use jsk_recognition overlay_color_to_mono
* Update CHANGELOG.rst for 0.8.0
* Contributors: Kei Okada, Kentaro Wada, Shingo Kitagawa, Yusuke Niitani, ban-masa, banmasa, pazeshun

0.8.0 (2016-05-31)
------------------
* add image jsk image_resizer
* fix failing remove gripper link from link-list, (member 'string' list) requries (member 'string' list #'equal)
* use objects-sib-boxes and coords inspite of objects-in-bin-boxes and coms
* use depth_registered for softkinetic_camera
* :try-to-pick-object use bbox for grasping
* add sib demo rviz
* Visualize target convex_hull published by RBO segmentation
* Add applying bin_contents hint node
* Add node to apply bin_contents hint to object recognition
* Add vgg16 object_recognition.launch
* add cpi decomposer in SIB
* Visualize segementation result in bin
* softkinetic_camera node respawn = true
* add image_proc/decimate
* deleted compressed target mask
* add dist and height visualizer
* segmentation_in_bin nodes continue to run when bin_info_array is not published
* Add picking-with-sib.l
* detect :inverse-kinematics nil return and avoid passing it to angle-vector
* remove duplicated rbo_segmentation_in_bin_node.py
* add apc2015 work_order test
* kinect2_torso launch use standalone complex nodelet
* Move publish_bin_info from sib to main.launch
* Use standalone_complexed_nodelet for setup_softkinetic.launch
* Fix typo in work_order.py
* Revert a part of `#1511 <https://github.com/start-jsk/jsk_apc/issues/1511>`_ thanks to `#1529 <https://github.com/start-jsk/jsk_apc/issues/1529>`_
* add max_weight param in work_order
* work_order sort consider graspability
* modify work_order to apply for apc2016
* work_order level3 check move to proper position
* remove unused arg JSON in sib launch
* move get_work_order and get_bin_contents func to jsk_apc2016_common
* baxter-interface.l : set joint-states-queue-size 2 for gripper and body, see https://github.com/jsk-ros-pkg/jsk_pr2eus/pull/229
* add rate param in work_order test
* Use rosrun for euslint checking
* work_order.py fix typo
* cherry-pick https://github.com/euslisp/jskeus/pull/380
* add assert in robot-model :inverse-kinmatics
* euslisp/jsk_2016_01_baxter_apc/baxter.l : change weight did not work well, need to remove gripper joint from link-list
* test-ik.l: add test check `#1470 <https://github.com/start-jsk/jsk_apc/issues/1470>`_
* revert a part of `#1525 <https://github.com/start-jsk/jsk_apc/issues/1525>`_, that genrate baxter.dae twice
* CMakeLists syntax fixed
* Avoid bug in robot-interface
* add baxter.urdf and baxter.dae target in CMakeLists
* add proper depends on baxter.xacro in CMakeLists
* Test two target coords in test-ik-in-bin for right gripper
* modify left hand self filter
* Don't send angle-vector if IK fails in approaching and lifting
* Use rotation-axis z in again approach
* fix position of softkinetic_camera
* add download script for test data
* test for sib using torso kinect
* wait before sib and remove needless move
* object world coords get into hash
* use segmentation_in_bin for both arm in main.l
* add y-axis angle of bin-overlook-pose
* Increase padding of right gripper in self_filter
* jsk_tools_add_shell_test supports from 2.0.14
* Increase padding of right gripper in self_filter
* Fix move-arm-body->order-bin to be slow
* Fix return_object for right arm
* add robot self filter to sib
* compress rbo mask image to point cloud size
* Fix view-hand-pose to be robust against gripper change
* Update softkinetic camera calibration files
* fix comment out in segmentation_in_bin.launch
* Fix padding of right gripper in self_filter
* Fix right gripper urdf not to use some stl files
* deleted arg INPUT_TARGET
* Fix return_object for right arm not to collide with bin top
* Fix padding of right gripper in self_filter
* Fix right gripper urdf not to use some stl files
* changed launch file to work with nodified sib
* Add rviz config for SIB visualization
* add a launch file that visualizes sib
* Remove tab in application-main.l
* Check tab in euslint
* add robot self filter for apc2016 robot model
* comment out right gripper self filter
* Suppress error output in IK defined in baxter-util.l
* Remove unofficial interface generators and old json files
* add robot self filter for apc2016 robot model
* add euslint test for every euslisp files
* Add euslint for euslisp source code
* fix bug in main.l
* change offset-from-entrance not to collide to bin top
* Approach objects straight down
* Don't overload gripper servo when placing object
* add publish target_bin
* Raise object height threshold of bending gripper
* split publish tf and publish bin info
* Tell heavy object from wall when using rarm
* Add roslaunch_add_file_check for vaccum_gripper.launch
* add :recognize-objects-segmentation-in-bin method
* Not to collapse vacuum pad
* func get_mask_img into one node
* equalize gripper length used in decision
* Add test of roslint for python
* Fix style of python code
* press gripper back against bin wall
* adjust offsets
* decide target end-coords depending on size of gripper and bin
* improve decision of approaching
* modify sib launch to use softkinetic camera
* separate segmentation_in_bin launch for each hand
* standarize POINTS -> CLOUD
* add :check-bin-exist method check if target bin is exist.
* deleted confusing setters
* exit from callback when target bin is false
* fix cmakelist depends path into full path
* fixed image format of message published by RBO_SIB node
* patch: change timestamp of the mask image from rbo_sib
* change launch file to call post-rbo process on sib
* Merge pull request `#1404 <https://github.com/start-jsk/jsk_apc/issues/1404>`_ from pazeshun/avoid-collision-body-arm-bin-h
  [jsk_2016_01_baxter_apc] avoid collision between body and rarm when pulling out rarm from Bin h
* add revert-if-fail arg for ik->bin-entrance
  this is for test-move-arm-to-bin
* Merge pull request `#1403 <https://github.com/start-jsk/jsk_apc/issues/1403>`_ from yuyu2172/sync-push
  [jsk_apc2016_common] add cpp message synchronizer
* softkinetic config modified
* remove unused tf broadcaster
* fix :rotation-axis from :t to t
* avoid collision between body and rarm when pulling out rarm from Bin h
* add cpp message synchronizer
* adjust lifting of objects when gripper is straight
* add header to rbo topic
* not to knock down objects
* add launch file that initiates segmentation_in_bin
* use rospy debug tools print -> rospy.loginfo
* change variables to get transform
* RBOSegmentationInBinNode inherits ConnectionBasedTransport
* add ik solution in bin test
* apply test to every bin
* Test move-arm-to-bin with rosbag for bin boxes
* 2015 launch files do not depend on 2016 config
* use apc2016 robot model
* add x-axis limit check
* clean up codes in ik-check.l
* add interface b/w segmentation_in_bin and ROS
* fix pass to and from bin e
* rotate gripper joint by script not by ik
* change middle point of ik
* don't use gripper joint to solve ik
* apply test to every bin
* Test move-arm-to-bin with rosbag for bin boxes
* Overwrite existing class names by managing loading order
* Succeed to reusable class for baxter interface/robot
* fixed arduino firmware to disable torque when serial_node.py is killed.
* use rotation of wrist to avoid ik failure
* add picking only code
* add recognition_in_hand and setup_torso
* use gripper servo off not to release objects
* change gripper servo angle case by case
* avoid collision between gripper and bin top
* remember angle-vector to pull out arm from bin
* return item closer to back of shelf
* Add enable_gripper node enabling grippers and managing automatic power off (`#1331 <https://github.com/start-jsk/jsk_apc/issues/1331>`_)
* interface-generator fixed into random (`#1333 <https://github.com/start-jsk/jsk_apc/issues/1333>`_)
* Fixed Arduino firmware (`#1335 <https://github.com/start-jsk/jsk_apc/issues/1335>`_)
* add softkinetic overlook pose method
* fix softkinetic position
* use new pressure sensor instead of one in vacuum cleaner
* fix default-controller of baxter-interface.l
* fix gripper_trajectory_server's info and extendability
* Remove pod model which is not used currently
* fix baxter-interface.l to move gripper servo when using rarm-controller
* output joint trajectory action server's stdout to screen
* add a velocity limit of gripper joint
* make lisp methods to power on/off gripper servo
* add LICENSE
* copy meshes and xacros from softkinetic_camera
* load depth calibration to setup_softkinetic.launch
* Update Calibration on 2016/04/20
  add depth calibration
* add right_gripper_vacuum_pad_joint to rarm chain in baxter.yaml
* use auto-generated baxter.l instead of baxter.l in baxtereus
* make yaml file for auto-generating
* add dependencies for generating baxter.l
* modify CMakeLists.txt to generate baxter.l
* add .gitignore to ignore auto-generated files
* fix work_order.py
* add test/work_order.test
* changed baxter.launch to run gripper_trajectory_server
* fixed CMakeLists.txt
* added gripper_trajectory_server.cpp
* fixed CMakeLists.txt
* added dependencies in package.xml
* fixed baxter.launch to launch gripper_joint_states_publisher
* added gripper_joint_states_publisher.cpp
* calibrate softkinetic 2016/04/17
* change softkinetic device
* modify camera serial number on right hand
* softkinetic image format fixed
* modify to launch softkinetic devices by serial number
* add right arm depthsense camera
* disable tweet
* change limits of right_gripper_vacuum_pad_joint
* add origin of the collision elements in urdf
* change robot name to baxter from baxter_creative
* change joint names of right gripper
* add old gripper to left arm
* add new right gripper
* add mesh data
* Contributors: Bando Masahiro, Kei Okada, Kentaro Wada, Shingo Kitagawa, Yusuke Niitani, Shun Hasegawa 

0.2.4 (2016-04-15)
------------------
* Rename launch file
* Fix typo left_vacuum_gripper.xacro
* Add softkinetic xacro
* Clean up setup_creative.launch
* Fix name right/left
* Rename camera to left_camera
* Rename setup_baxter_gazebo -> initialize_baxter
* Initialize docs for 'jsk_2016_01_baxter_apc'
* Get organized point cloud from softkinetic camera
* Chang file name
* Add urdf model of Baxter with creative on right hand
* Change baudrate to 115200
* Change jsk_2015_05_baxter_apc/urdf/ -> jsk_2015_05_baxter_apc/robots/
* Add baxter.launch and new arduino node
* Chang topic name
* Add servo state controller in arduino firmware
* Enable to control servo with ros
* Add arduino nano firmware
* Contributors: Kentaro Wada, Shingo Kitagawa, Yusuke Niitani, Masahiro Bando, Shun Hasegawa

0.2.3 (2016-04-11)
------------------
* Bugfix
  + Add jsk_2015_05_baxter_apc as build_depend to fix `#1253 <https://github.com/start-jsk/jsk_apc/issues/1253>`_
  + Fix catkin config of 'jsk_2016_01_baxter_apc'
  + Add jsk_2015_05_baxter_apc as run_depend
  + Generate euslisp manifest
* Migration
  + Move urdf/ -> robots
  + Copy 'euslisp/main.l' from jsk_2015_05_baxter_apc
  + Rename 'euslisp/main.l' -> 'euslisp/application-main.l'
  + Copy 'euslisp/jsk_2015_05_baxter_apc' from jsk_2015_05_baxter_apc
  + Copy 'launch/main.launch' from jsk_2015_05_baxter_apc
* Visualization
  + visualize rect, label and proba
* Interface
  + add blacklist for bin contents
* Task Process
  + Do not trust pressure sensors based detection of grasps
  + Skip Level3 work_order (number of bin_contetns > 5)
* Motion Test
  + reset pose before ik check
  + add ik-check.l and ik-range.l for checking IK-solvable range in bins
  + use pushback and fix and add dump file save line
* Documentation
  + Fix readme title jsk_apc -> jsk_2016_01_baxter_apc
* Misc
  + modified not to update already generated JSON
  + interface_generator cleaned up into class
  + rename json file
  + interface_generator modified for apc2016
* Contributors: Heecheol Kim, Kentaro Wada, Shingo Kitagawa, Yusuke Niitani

0.2.2 (2016-03-08)
------------------

0.2.1 (2016-03-08)
------------------
* fix maintainer/author in package.xml
* Contributors: Kei Okada

0.2.0 (2016-03-08)
------------------
* Try APC2016 with program for APC2015
  * Json file for picking: Layout1
  * Add Shared files for jsk_2016_01_baxter_apc
    Modified:
    - jsk_2016_01_baxter_apc/README.md
* Semi for 2015B4
  * apc_pick modified
  * json files for simulation added
  * documentation added
  * add interface_generator
  * [2016 apc] rename launch file
  * change baxter software version
  * rm json file
  * stow recognition modified
  * [2016apc] modify stow recognition launch file
  * [2016apc] modify viz-recog.l and add json
  * [2016apc] add stow recognition launch
  * [2016apc] add visualize program for recognition
  * Add kinect2_external
  * [2016 apc] modify stow_kiva.world.erb for stow tasks
  * add initial camera position
  * add kiva_stow and baxter_stow
  * add ruby to build depend for erb
  * add jsk_2016_01_baxter_apc
* Contributors: Heecheol Kim, Kei Okada, Kentaro Wada, Shingo Kitagawa, Masahiro Bando
