^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_2016_01_baxter_apc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.1.4 (2017-10-15)
------------------

4.1.3 (2017-10-12)
------------------

4.1.2 (2017-10-11)
------------------

4.1.1 (2017-10-10)
------------------
* add object-index key in pick-object-in-order-bin
* use avs-raw with :fast in pick-object
* update shelf marker
* Fix missing include files
  I got `No such file or directory "ros/ros.h"`.
* Move installation for jsk_2016_01_baxter_apc to doc
* Add light meshes
* Remove unnecessary meshes
* Contributors: Kentaro Wada, Shingo Kitagawa, Shun Hasegawa

4.1.0 (2017-08-12)
------------------
* use transformable_markers for kiva_pod
* move camera launch in setup launch
* Update LICENSE
* Move library to euslisp/lib for jsk_2015_05_baxter_apc
* Move library to euslisp/lib for jsk_2016_01_baxter_apc
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
* Use baxter_simple.urdf in jsk_2016_01_baxter_apc baxterrgv5.xacro
* Use baxter_simple.urdf in jsk_2016_01_baxter_apc baxter.xacro
* Re enable jsk_2016_01_baxter_apc's tests
* Contributors: Kentaro Wada

3.3.0 (2017-07-15)
------------------
* disable test to pass travis
* Contributors: Kei Okada

3.2.0 (2017-07-06)
------------------

3.1.0 (2017-06-30)
------------------
* Change save_dir in dynamic
* Launch right stereo camera in baxter.launch
* move stereo_camera_info files from jsk_2016_01_baxter_apc to jsk_arc2017_baxter
* Use arduino symlink in vacuum gripper
* Order agonistic options to control vacuum gripper
* Don't set link-list as nil in IK when it is unspecified (`#2134 <https://github.com/start-jsk/jsk_apc/issues/2134>`_)
* remove unused include_dirs
* move ik->nearest-pose method to baxter.l
* fix camera_name in yaml files
* add calibration yaml files of astra mini stereo
* temporarily tuned pressure threshold in calibration
* Contributors: Kentaro Wada, Shingo Kitagawa, Shun Hasegawa, Yuto Uchimi, YutoUchimi

3.0.3 (2017-05-18)
------------------
* fix default-controller not to set head twice (`#2097 <https://github.com/start-jsk/jsk_apc/issues/2097>`_)
* add roseus_mongo as run_depend (`#2099 <https://github.com/start-jsk/jsk_apc/issues/2099>`_)
* Contributors: Shingo Kitagawa

3.0.2 (2017-05-18)
------------------
* Put in order tags in CHANGELOG.rst
* Fix torque limit of prismatic joint in gripper (`#2094 <https://github.com/start-jsk/jsk_apc/issues/2094>`_)
* Move astra_hand_rgv5.launch into baxterrgv5.launch
* Contributors: Kentaro Wada, Shun Hasegawa

3.0.1 (2017-05-16)
------------------
* Find automatically astra device_id
* mv euslint to jsk_apc2016_common package
* Contributors: Kentaro Wada, Shingo Kitagawa

3.0.0 (2017-05-08)
------------------
* fix syntax in euslint
* improve euslint to accept path
* fix too long line in baxter-interface
* modify min-pressure threshold
* rename movable-region\_ and set it in slots
* add head-controller in rarm-controller
* rename jsk_2016_01_baxter_apc slots name
* modify object_segmentation_3d to accept args
* Fix rosdep key for gdown: python-gdown-pip
* Include astra_hand in astra_hand_rgv5
* Add main launch for baxterrgv5
* Add lisp main and examples for baxterrgv5
* Add baxter-interface of baxterrgv5
* Calibrate Astra for gripper-v5
* Add launch for baxterrgv5
* Add ros_control system for gripper-v5
* Add launch for gripper-v5 dynamixel controllers
* Add a dynamixel controller for gripper-v5
* Add hose_connector_manager firmware
* Add models of baxterrgv5 (baxter with right gripper-v5)
* Add gripper-v5 meshes
* merge baxter_sim.launch and include/baxter_sim.xml
* Use rqt_yn_btn instead of rviz plugin
  In order to avoid the blocking of topic update on rviz.
* Add link to wiki
* Merge pull request `#2030 <https://github.com/start-jsk/jsk_apc/issues/2030>`_ from pazeshun/fix-dup-def
  Fix duplicate definition of variables
* Enable test_move_arm_to_bin
* Fix duplicate definition of variables
* Skip bin_B for which ik is difficult to be solved
* Faster scale for data collection
* update CMakelists to simplify euslint test
* fix typo in baxter_pick_sim and baxter_stow_sim
* add baxter_sim launch
* disable collision between head and gripper
* disable collision between pedestal and gripper
* disable collision between display and screen
* remove unused rosinstall
* Fix tf rate of camera 10 -> 100
* Use - instead of _ to define variable further
* Use - instead of _ to define variable
* Use let instead of let* as much as possible
* euslint escape ; in quotation
* Visualize objects in bin on Euslisp
* add left_first args to setup_for_stow
* Merge pull request `#1994 <https://github.com/start-jsk/jsk_apc/issues/1994>`_ from pazeshun/add-rotate-wrist-ik
  Add :rotate-wrist-ik
* Add :rotate-wrist-ik
* Fix long and bad variable name
* Fix :ik->nearest-pose not to move arms unintendedly
* Merge pull request `#1999 <https://github.com/start-jsk/jsk_apc/issues/1999>`_ from knorth55/ik-check-improve
  fix ik-check to work proper with gripper-v2
* fix codes in ik-check
* use gripper and set rotation axis in ik-check
* use alist to avoid segmentation fault (`#1997 <https://github.com/start-jsk/jsk_apc/issues/1997>`_)
* consider object bounding box to pick from tote
* Enable to set ik-prepared-poses
* Merge pull request `#1985 <https://github.com/start-jsk/jsk_apc/issues/1985>`_ from pazeshun/not-move-gripper
  Don't move gripper while carrying object in picking-with-sib
* Don't move gripper while carrying object in picking-with-sib
* Add no-gripper-controller
* Enable move-arm-body->order-bin to set controller type
* Enable send-av to set controller type
* Adjust picking-with-sib to current object segmentation
* Add method to get the work order of certain bin
* add gazebo material tag for gripper
* add inertia tag in both gripper.urdf.xacro
* fix trajectory_execution namespace
  see https://github.com/ros-planning/moveit/issues/61
* mv LICENSE and add kiva_pod stl model
* add ompl_planning and update param
  longest_valid_segment_fraction: 0.05 -> 0.01
  this solves moveit path simplification error.
* specify to use av-seq-raw in spin-off-by-wrist
* add moveit group "both_arms" as :arms
* refine pick-object-in-order-bin motion
* fix movable-region warning
  current: this warning always shows up
* remove wait-interpolation from :hold-opposite-hand-object
* add dy key in :view-opposite-hand-pose
* add distance :move-arm-body->bin-with-support-arm
* remove place-object-pose when picking from tote
* modify view-hand-pose because we don't use kinect2
* add data_collection launch
* add moveit-environment in baxter-interface
  default key :moveit nil
  if you want to enable moveit, you need to set key :moveit t.
  (jsk_2016_01_baxter_apc::baxter-init :moveit t)
* add baxter_moveit launch for moveit usage
* add jskbaxter2 moveit_config
* add gazebo tag in vacuum_gripper.xacro
* set nil not to initialize default moveit config
* add gripper_trajectory_server for simulator
* update xacro wiki url
* Fix position of arduino firmware
* Add urdf checking launch
* comment out vgg object verification node
* Fix for not working :interpolatingp on simulation
  - See :wait-interpolation on pr2eus/robot-interface.l also.
* add :move-arm-body->bin-with-support-arm in baxter-interface
* add :hold-opposite-hand-object in baxter-interface
* add :approaching-from-downside-pose in baxter-robot
* add :view-opposite-hand-pose in baxter-robot
* Fix typo in tmuxinator config
* add wait-interpolation-until-grasp method
* add option in euslint and remove indent check
* Add config for tmuxinator
* Add missing run_depend
* Adjust right hand mounted astra camera
* Fix KeyError for bin without target object
* Support no target in rqt_select_target
* modify debug-view nil not to show debug log
* comment out drawing irtviewer line
* Move images under jsk_apc2016_common to use it in launch correctly
* Remove check_baxter_pkg_version.sh that is not used
  You can just run in shell:
  ```
  rospack list | awk '{print $1}' | grep baxter | xargs -t -n1 rosversion
  ```
* Remove old README from jsk_2016_01_baxter_apc
  See https://github.com/start-jsk/jsk_apc#install
* Move srv to common package to fix dependency graph
  - dependency graph should be jsk_2016_01_baxter_apc -> jsk_apc2016_common
* Contributors: Kentaro Wada, Shingo Kitagawa, Shun Hasegawa, pazeshun

2.0.0 (2016-10-22)
------------------
* fix error Unknown limb is passed: :arms
* format to pass test work_order_server
* rqt_select_target use service to update work_order
* rosparam pass work_order bin_contents from json
* use class and rospy.timer for work_order_server
* rename work_order.py -> work_order_server.py
* rename node: work_order -> strategic_work_order
* Merge pull request `#1895 <https://github.com/start-jsk/jsk_apc/issues/1895>`_ from knorth55/param-contents
  use param to pass bin contents and tote contents
* Merge pull request `#1896 <https://github.com/start-jsk/jsk_apc/issues/1896>`_ from start-jsk/fit-apply-context-to-label-probability
  Fit to apply_context_to_label_proba which is merged to jsk_perception
* Add yes_no_button for user input
* Launch rviz for user input in main.launch
* use rosparam to pass bin_contents
* change set_tote_contents_param to json_to_rosparam
* Use yes_no_button panel in rviz for user input
* Slower fold-pose-back initialization for apc task
* Add method to set object segmentation candidates to ri
* Fit to apply_context_to_label_proba which is merged to jsk_perception
* add json utils in util.l
* fix apc2016 simulation for baxter_simulator v1.2
* use arm2str in baxter-interface
* use object_segmentation_3d launch for stow task
* update tote pose
* Set initial target bin to check sanity
* Add checking sanity script for setup_for_pick.launch
* Add rviz config for pick demo
* Remove no need nodes from main.launch
* Use new 3D object segmentaion pipeline with euslisp controller
* Introduce new 3D object segmentation pipeline
  As proposed in https://github.com/start-jsk/jsk_apc/issues/1865
* Support non-list arg for ros::set-dynparam
* Add arm2str as util and use it
* Skip verification because of its unreliability
* Calibrate extrinsic parameters of astra cameras
* add astra intrinsic calibration file
* Add args to astra_hand.launch
* Add calibboard stickers
* Calibrate right hand mounted camera depth
  Also this updates the rvizconfig
  Conflicts:
  jsk_2016_01_baxter_apc/launch/include/astra_hand.launch
* add calib-pressure option in main program
* Use nodes to test arm-to-bin motion instead of rosbag
* Publish bin bounding boxes in baxter.launch
  This is useful because we can use baxter-interface.l without main.launch or main_stow.lauch.
* CMakeLists.txt: need to set current directory to ROS_PACKAGE_PATH
* Merge pull request `#1871 <https://github.com/start-jsk/jsk_apc/issues/1871>`_ from knorth55/test-stow-work-order
  add stow_work_order test
* add stow_work_order test
* add stow_work_order option not to output json
* add ik test in tote for stow task
* Adjust right gripper firmware to gripper-v4
* minor fix for real robot
* use protected/private for variable
* add minjerk class
* include pub/sub within c++ object
* use Object Oriented callback style
* Adjust astra hand
* Adjust calib pressure threshold again
* minjerk and continuous feedback
* depth register works with explicit arg
* Contributors: Kei Okada, Kentaro Wada, Shingo Kitagawa, Yusuke Niitani, pazeshun

1.5.1 (2016-07-15)
------------------
* Get lower pressure threshold in :calib-pressure-threshold
  By changing the subtraction value from 8 to 10.
* Set minimum pressure as the threshold for no_object
* Adjust calib-pressure-threshold for real gripper
* Remove no need condition in update-pressure-threshold
* 1.5.0
* Update CHANGELOG.rst to release 1.5.0
* rotate gripper after picking object from tote
* Fix bug in FCNMaskForLabelNames because of mask image value
* fix typo in dropped detection
* fix typo in dropped detection
* json update msg improved
* improve volume_first work order
* rotate gripper in bin
* Add apply mask to get reachable space image
* Fix type to find contour with cv2
* Draw contour to remove big object cleanly
* Fix some bugs in fcn_mask_for_label_names.py
* Fix launch files for removeing big object in tote
* Fix typo in fcn_mask generation code
* Fix typo
* Launch fcn node in boa
* Add feature to remove cloud of blacklist objects for stow task
* clear params for blacklisted object
* add info and warn for dropped while place in bin
* listed out all blacklisted object
* servo on when return from bin
* servo on before view hand pose
* detect dropped object in place_object andnot update json
* modify json update duration
* Skip target_bin is empty in ouptut_json_stow.py
* Fix typo in main-stow.l
* add offset in pick-object in -order bin
* fix rotation of in tote clipper
* add dr_browns_bottle_brush in blacklist
* improve stow motion
* add no_object in apply_tote_contents_hint
* Fix typo in apply_tote_contents_hint.py
* add blacklist in apply_tote_contents_hint
* get smaller movable region
* Enhance ros-info for recognized object in hand
* Longer timeout for in-hand-object-recognition in main-stow.l
* add need-to-wait condition
* change motion of removing arm from order bin
* modify in hand clipper size
* fix bug in select target-bin
* if theres is no proper target-bin, use random target-bin
* increase object length
* Visualize rosconsole of euslisp main script
* Show node name in ros-info
* increase volume limit
* z offset modified to APC2016 real kiva
* use object length view pose
* add blacklist object returning back to tote
* rename black_list to volume_first
* adjust tote for APC2016
* remove head controller for rarm
* add head-controller
* use fixed offset
* not use euclid clustering
* in hand clipper modified
* rotate gripper when exiting from bin
* avoid arm collision with head
* remove no_object label in apply_tote_contents_hint
* fix apply_tote_contents_hint
* use work-order msg for :select-stow-target-bin
* add stow_work_order_server node
* recognize object in hand and verify
* add no_object candidates in apply_tote_contents_hint
* fix path in vgg16_object_recognition launch
* add calib-pressure-threshold in stow main program
* add node for output stow json
* add in hand recognition for stow task launch
* enable visualize stow json
* remove self filter in recognition_in_hand_for_stow
* fix bug in :cube->movable-region
* fix random object-index to pick same object in pick-object-in-order-bin
* blacklist bin :l for large object stow task
* modify order-bin-overlook-pose
* fix typo in need-to-wait-opposite-arm
* if fail-count > 1, wait opposite arm start picking
* add ros-info in return_from_bin in stow main
* set boundary of tote for y axis
* add wait condition for pick_object in stow task
* modify order bin overlook pose
* get into wait_for_opposite_arm_in_pick after pick fail
* recognize object length after detecting graspingp
* modify view hand pose for stow-task
* stop-grasp if there is no object in view hand pose
* trust pressure sensor in stow main program
* set movable-region to avoid arm from moving tote
* add recognize-order-bin-box
* remove unused nodes from segmentation_each_object_in_tote
* add more condition for need-to-wait-opposite-arm
* wait opposite arm in place condition added
* get graspingp after second approach
* add gripper-servo-on before approaching to object
* picking from tote n-trial 3 -> 2
* Revert "bin :e blacklisted because of dangerous move"
  This reverts commit b86f4374d3210823ef7801e4084c842a295de1f6.
* pick object randomly from tote
* add wait-opposite-arm when returning from bin
* combine all wait-oppsite-arm-for-stow method to one
* use satan for vgg16 in stow task
* fix line length < 100 to pass run_tests
* use different attention clipper for each arm
* use astra for segmentation_in_tote
* no more use for self filter
* modify object length limit to 1.0 and take longer timeout
* bin :e blacklisted because of dangerous move
* fix clipper for gripper v3
* rename set_bin_param -> publish_bin_info for stow main
* use proper bin for entering large object
* rotate gripper to 45 when entering large object
* rotate gripper to 0 and use lower traj for exit
* if object length > 0.2, use higher traj and put further
* add publish_bin_bbox for stow task
* use avoid-shelf-pose instead of move-arm-body->bin to avoid quick move
* add scale key in move-arm-body->bin
* add SupervoxelSegmentation for picking from tote
* fix bug in object length method
* add object length recognize method and use it in stow
* use gripper v3 for in_hand_clipper
* add wait opposite arm for place object and pick object
* use gripper v3 for left arm in stow main program
* add vgg16 node for stow task
* add inside tote recognition launch and connect to main program
* add euclid clustering in tote for stow task
* add stow task main program and launch file
* add stow method and slots in baxter-interface.l
* Adjust astra_hand camera
* Add fcn trained data to download
* 1.0.0
* Update CHANGELOG.rst
* Fix for pep8
* Fix for euslint
* Revert "Enhance :view-hand-pose for each bin"
  This reverts commit 4949769c068829e4a490f5cb007545578c17727e.
* Revert "Revert view-hand-pose for bin :g :h :i"
  This reverts commit 708196580f5bd1f2e54fe2ef99669f4df70d6434.
* Add feature to skip verification in main.l
* Show visualize json on xdisplay in main.launch
* astra calibration
* Fix pressure threshold
* Fix return_object
* Rotate gripper earlier in drawing out arm
* Fix return_object to avoid collision between body and arm
* Fix offset-gripper-bin-side
* Fix offset of return_object
* Lift object to world-z in side approaching
* Fix offset of object width
* Fix timing of rotating grippers
* Change gripper-angle not to draw out objects
* Change gripper-angle not to push target object
* Lift object higher
* Enhance main.l for logging
* Avoid collision between gripper and bin side wall
* Improve return_object not to drop
* Enhance ros-info in main.l and baxter-interface.l
* Fix typo for data collection in main.launch
* Fix typo in data collection
* Remove no need debug printing in baxter-interface.l
* Add no_object label as candidate for picking
* Enhance the logging in :verify-object with green color
* Stop grasp when graspingp is nil in verify_object
* Fix bug of deciding object depth
* Fix offset of object height
* change launch to handle debug output
* change fcn launch file to use depth img
* Show recognition result as green
* Fix bug of ik->bin-entrance
* Set queue_size=1 for apply_bin_contents_hint.py
* Add tools for euslisp to log info with color
* astra camera calib
* Improve view-kiva-pose
* Data collection program in hand while apc main loop
* Gripper servo on after user input
* Change initial pose to view-kiva-pose
* Fix return_object not to drop
* visualize rosinfo output of main.l on rviz
* Set graspingp after avoid-shelf-pose
* Decrease segmentation in bin timeout
* Set rosparam at the top of state in main.l
* Stop vacuum when e-stop is pressed
* baxter-interface.l : remove head-controller from defaut-controller ( @pazeshun I think we should not change :rarm-contller instaed, we should use rarm-head-controller, or when there is :ctype :rarm-controller, then we add :head-controller
* Remove abanding strategy for level3
* Add avoid-shelf-pose for safety and skip verification if number of bin contents is 1
* Feature to abandon work_order by user requests
* Change bin reachable depth
* Get deep object with shallow hand position
* Add bin-reachable-depth method and use it
* Make aborting by depth safe
* Change object-found-p to local variable
* Use keep-picking-p in main.l
* Add keep-picking-p method
* Change variable name is-object-found -> object-found-p
* Add offset of object width to decide approach direction
* Fix typo of offset
* Revert view-hand-pose for bin :g :h :i
* Enhance :place_object in order not to drop object
* Fix offset
* Remove checking grasps in :verify_object state
* Enhance :view-hand-pose for each bin
* Prevent collision between gripper camera and bin
* Add script to check ik-bin-entrance
* Change hardcoded pose in baxter-interface
* Fix typo of main.l
* Fix typo in baxter-interface
* astra hand calib
* Fix ik->bin-entrance not to fail when gripper-angle is 0
* Apply offset to pick object's center
* Change main.l to use recognize-objects-in-bin-loop
* Add recognize-objects-in-bin-loop method
* Add bin-overlook-pose method
* Prevent IK fail when drawing out arm
* Set rthre as 10 degree
* Return object when graspingp nil
* Use object_data in work_order.py
* Adjust move-arm-body->bin-overlook-pose for APC final
* Add script to test bin-overlook-pose
* Skip objects whose graspability exceeds threshold 3
* Fix :verify_object mode in main.l
* Add fold-pose-back.l script
* Adjust left astra hand camera
* Update check_astra.rviz
* Adjust right astra hand camera
* Remove subscribing topic for visualization on rviz
  For computational loss.
* Merge pull request `#1838 <https://github.com/start-jsk/jsk_apc/issues/1838>`_ from wkentaro/set-dynparam-eus
  Set dynamic reconfigure parameters in euslisp node
* Use ros::set-dynparam in in-hand-data-collection.l
* Set dynamic reconfigure parameters in euslisp node
* Merge pull request `#1831 <https://github.com/start-jsk/jsk_apc/issues/1831>`_ from wkentaro/longer-verify
  Longer timeout for vgg16 object recognition
* Merge pull request `#1817 <https://github.com/start-jsk/jsk_apc/issues/1817>`_ from pazeshun/not-need-nil-list
  Set nil instead of list when no object found
* Remove no_object label in apply_bin_contents_hint.py to trust pressure
* Longer timeout for vgg16 object recognition
* Merge pull request `#1792 <https://github.com/start-jsk/jsk_apc/issues/1792>`_ from yuyu2172/stop-self-filter
  stop using self filters
* Change overlook pose by @yuyu2172
* launch that visualizes fcn class label
* wait longer before starting to subscribe to sib result
* Set nil instead of list when no object found
* Fix memory leak in apply_bin_contents_hint.py
* add fcn launch file
* segmentation_in_bin.launch does not launch sib node
* Calibrate grasps in in-hand-data-collection-main.l
* Merge pull request `#1807 <https://github.com/start-jsk/jsk_apc/issues/1807>`_ from pazeshun/fix-overlook-pose
  Fix bin-overlook-pose
* Erase previous SIB data when SIB fails
* Fix bin-overlook-pose
* changed do-stop-grasp t
* Rolling gripper on closer point to robot
* Make data collection in main.launch as optional
* add collect sib data in main.launch
* move collect sib to launch/include
* collect sib data more modular
* Add no_object label in apply_bin_contents
* Fix bug of arm variable
* fixed firmware to use toggle switch
* Make :ik->nearest-pose method
* Data collection program for segmentation in bin
* Merge pull request `#1793 <https://github.com/start-jsk/jsk_apc/issues/1793>`_ from ban-masa/auto-pressure-calib
  Auto calib pressure threshold
* Use mask image to enhance object recognition result with vgg16 net
* added calib-pressure-threshold
* Prepare for logging
* Use VGG16 net for APC2016 in recognition_in_hand.launch
* Align bounding boxes to robot base frame
* stop using self filter
* Fix position of wait-interpolation-smooth
* Remove :recognize-objects-in-bin in picking-with-sib.l
* Merge pull request `#1784 <https://github.com/start-jsk/jsk_apc/issues/1784>`_ from pazeshun/abort-approach-ik-fail
  Abort picking objects when IK to it fails
* add use-current-pose in ik->bin-entrance
* improve ik->bin-entrance to minimize norm
* 0.8.1
* update CHANGELOG
* 0.8.1
* add roslint to package.xml
* update maintainers
* Abort picking objects when IK to it fails
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
* Contributors: Bando Masahiro, Kei Okada, Kentaro Wada, Shingo Kitagawa, Yusuke Niitani, ban-masa, banmasa, pazeshun

1.5.0 (2016-07-09)
------------------
* rotate gripper after picking object from tote
* Fix bug in FCNMaskForLabelNames because of mask image value
* fix typo in dropped detection
* fix typo in dropped detection
* json update msg improved
* improve volume_first work order
* rotate gripper in bin
* Add apply mask to get reachable space image
* Fix type to find contour with cv2
* Draw contour to remove big object cleanly
* Fix some bugs in fcn_mask_for_label_names.py
* Fix launch files for removeing big object in tote
* Fix typo in fcn_mask generation code
* Fix typo
* Launch fcn node in boa
* Add feature to remove cloud of blacklist objects for stow task
* clear params for blacklisted object
* add info and warn for dropped while place in bin
* listed out all blacklisted object
* servo on when return from bin
* servo on before view hand pose
* detect dropped object in place_object andnot update json
* modify json update duration
* Skip target_bin is empty in ouptut_json_stow.py
* Fix typo in main-stow.l
* add offset in pick-object in -order bin
* fix rotation of in tote clipper
* add dr_browns_bottle_brush in blacklist
* improve stow motion
* add no_object in apply_tote_contents_hint
* Fix typo in apply_tote_contents_hint.py
* add blacklist in apply_tote_contents_hint
* get smaller movable region
* Enhance ros-info for recognized object in hand
* Longer timeout for in-hand-object-recognition in main-stow.l
* add need-to-wait condition
* change motion of removing arm from order bin
* modify in hand clipper size
* fix bug in select target-bin
* if theres is no proper target-bin, use random target-bin
* increase object length
* Visualize rosconsole of euslisp main script
* Show node name in ros-info
* increase volume limit
* z offset modified to APC2016 real kiva
* use object length view pose
* add blacklist object returning back to tote
* rename black_list to volume_first
* adjust tote for APC2016
* remove head controller for rarm
* add head-controller
* use fixed offset
* not use euclid clustering
* in hand clipper modified
* rotate gripper when exiting from bin
* avoid arm collision with head
* remove no_object label in apply_tote_contents_hint
* fix apply_tote_contents_hint
* use work-order msg for :select-stow-target-bin
* add stow_work_order_server node
* recognize object in hand and verify
* add no_object candidates in apply_tote_contents_hint
* fix path in vgg16_object_recognition launch
* add calib-pressure-threshold in stow main program
* add node for output stow json
* add in hand recognition for stow task launch
* enable visualize stow json
* remove self filter in recognition_in_hand_for_stow
* fix bug in :cube->movable-region
* fix random object-index to pick same object in pick-object-in-order-bin
* blacklist bin :l for large object stow task
* modify order-bin-overlook-pose
* fix typo in need-to-wait-opposite-arm
* if fail-count > 1, wait opposite arm start picking
* add ros-info in return_from_bin in stow main
* set boundary of tote for y axis
* add wait condition for pick_object in stow task
* modify order bin overlook pose
* get into wait_for_opposite_arm_in_pick after pick fail
* recognize object length after detecting graspingp
* modify view hand pose for stow-task
* stop-grasp if there is no object in view hand pose
* trust pressure sensor in stow main program
* set movable-region to avoid arm from moving tote
* add recognize-order-bin-box
* remove unused nodes from segmentation_each_object_in_tote
* add more condition for need-to-wait-opposite-arm
* wait opposite arm in place condition added
* get graspingp after second approach
* add gripper-servo-on before approaching to object
* picking from tote n-trial 3 -> 2
* Revert "bin :e blacklisted because of dangerous move"
  This reverts commit b86f4374d3210823ef7801e4084c842a295de1f6.
* pick object randomly from tote
* add wait-opposite-arm when returning from bin
* combine all wait-oppsite-arm-for-stow method to one
* use satan for vgg16 in stow task
* fix line length < 100 to pass run_tests
* use different attention clipper for each arm
* use astra for segmentation_in_tote
* no more use for self filter
* modify object length limit to 1.0 and take longer timeout
* bin :e blacklisted because of dangerous move
* fix clipper for gripper v3
* rename set_bin_param -> publish_bin_info for stow main
* use proper bin for entering large object
* rotate gripper to 45 when entering large object
* rotate gripper to 0 and use lower traj for exit
* if object length > 0.2, use higher traj and put further
* add publish_bin_bbox for stow task
* use avoid-shelf-pose instead of move-arm-body->bin to avoid quick move
* add scale key in move-arm-body->bin
* add SupervoxelSegmentation for picking from tote
* fix bug in object length method
* add object length recognize method and use it in stow
* use gripper v3 for in_hand_clipper
* add wait opposite arm for place object and pick object
* use gripper v3 for left arm in stow main program
* add vgg16 node for stow task
* add inside tote recognition launch and connect to main program
* add euclid clustering in tote for stow task
* add stow task main program and launch file
* add stow method and slots in baxter-interface.l
* Adjust astra_hand camera
* Add fcn trained data to download
* Contributors: Kentaro Wada, Shingo Kitagawa

1.0.0 (2016-07-08)
------------------
* Fix for pep8
* Fix for euslint
* Revert "Enhance :view-hand-pose for each bin"
  This reverts commit 4949769c068829e4a490f5cb007545578c17727e.
* Revert "Revert view-hand-pose for bin :g :h :i"
  This reverts commit 708196580f5bd1f2e54fe2ef99669f4df70d6434.
* Add feature to skip verification in main.l
* Show visualize json on xdisplay in main.launch
* astra calibration
* Fix pressure threshold
* Fix return_object
* Rotate gripper earlier in drawing out arm
* Fix return_object to avoid collision between body and arm
* Fix offset-gripper-bin-side
* Fix offset of return_object
* Lift object to world-z in side approaching
* Fix offset of object width
* Fix timing of rotating grippers
* Change gripper-angle not to draw out objects
* Change gripper-angle not to push target object
* Lift object higher
* Enhance main.l for logging
* Avoid collision between gripper and bin side wall
* Improve return_object not to drop
* Enhance ros-info in main.l and baxter-interface.l
* Fix typo for data collection in main.launch
* Fix typo in data collection
* Remove no need debug printing in baxter-interface.l
* Add no_object label as candidate for picking
* Enhance the logging in :verify-object with green color
* Stop grasp when graspingp is nil in verify_object
* Fix bug of deciding object depth
* Fix offset of object height
* change launch to handle debug output
* change fcn launch file to use depth img
* Show recognition result as green
* Fix bug of ik->bin-entrance
* Set queue_size=1 for apply_bin_contents_hint.py
* Add tools for euslisp to log info with color
* astra camera calib
* Improve view-kiva-pose
* Data collection program in hand while apc main loop
* Gripper servo on after user input
* Change initial pose to view-kiva-pose
* Fix return_object not to drop
* visualize rosinfo output of main.l on rviz
* Set graspingp after avoid-shelf-pose
* Decrease segmentation in bin timeout
* Set rosparam at the top of state in main.l
* Stop vacuum when e-stop is pressed
* baxter-interface.l : remove head-controller from defaut-controller ( @pazeshun I think we should not change :rarm-contller instaed, we should use rarm-head-controller, or when there is :ctype :rarm-controller, then we add :head-controller
* Remove abanding strategy for level3
* Add avoid-shelf-pose for safety and skip verification if number of bin contents is 1
* Feature to abandon work_order by user requests
* Change bin reachable depth
* Get deep object with shallow hand position
* Add bin-reachable-depth method and use it
* Make aborting by depth safe
* Change object-found-p to local variable
* Use keep-picking-p in main.l
* Add keep-picking-p method
* Change variable name is-object-found -> object-found-p
* Add offset of object width to decide approach direction
* Fix typo of offset
* Revert view-hand-pose for bin :g :h :i
* Enhance :place_object in order not to drop object
* Fix offset
* Remove checking grasps in :verify_object state
* Enhance :view-hand-pose for each bin
* Prevent collision between gripper camera and bin
* Add script to check ik-bin-entrance
* Change hardcoded pose in baxter-interface
* Fix typo of main.l
* Fix typo in baxter-interface
* astra hand calib
* Fix ik->bin-entrance not to fail when gripper-angle is 0
* Apply offset to pick object's center
* Change main.l to use recognize-objects-in-bin-loop
* Add recognize-objects-in-bin-loop method
* Add bin-overlook-pose method
* Prevent IK fail when drawing out arm
* Set rthre as 10 degree
* Return object when graspingp nil
* Use object_data in work_order.py
* Adjust move-arm-body->bin-overlook-pose for APC final
* Add script to test bin-overlook-pose
* Skip objects whose graspability exceeds threshold 3
* Fix :verify_object mode in main.l
* Add fold-pose-back.l script
* Adjust left astra hand camera
* Update check_astra.rviz
* Adjust right astra hand camera
* Remove subscribing topic for visualization on rviz
  For computational loss.
* Merge pull request `#1838 <https://github.com/start-jsk/jsk_apc/issues/1838>`_ from wkentaro/set-dynparam-eus
  Set dynamic reconfigure parameters in euslisp node
* Use ros::set-dynparam in in-hand-data-collection.l
* Set dynamic reconfigure parameters in euslisp node
* Merge pull request `#1831 <https://github.com/start-jsk/jsk_apc/issues/1831>`_ from wkentaro/longer-verify
  Longer timeout for vgg16 object recognition
* Merge pull request `#1817 <https://github.com/start-jsk/jsk_apc/issues/1817>`_ from pazeshun/not-need-nil-list
  Set nil instead of list when no object found
* Remove no_object label in apply_bin_contents_hint.py to trust pressure
* Longer timeout for vgg16 object recognition
* Merge pull request `#1792 <https://github.com/start-jsk/jsk_apc/issues/1792>`_ from yuyu2172/stop-self-filter
  stop using self filters
* Change overlook pose by @yuyu2172
* launch that visualizes fcn class label
* wait longer before starting to subscribe to sib result
* Set nil instead of list when no object found
* Fix memory leak in apply_bin_contents_hint.py
* add fcn launch file
* segmentation_in_bin.launch does not launch sib node
* Calibrate grasps in in-hand-data-collection-main.l
* Merge pull request `#1807 <https://github.com/start-jsk/jsk_apc/issues/1807>`_ from pazeshun/fix-overlook-pose
  Fix bin-overlook-pose
* Erase previous SIB data when SIB fails
* Fix bin-overlook-pose
* changed do-stop-grasp t
* Rolling gripper on closer point to robot
* Make data collection in main.launch as optional
* add collect sib data in main.launch
* move collect sib to launch/include
* collect sib data more modular
* Add no_object label in apply_bin_contents
* Fix bug of arm variable
* fixed firmware to use toggle switch
* Make :ik->nearest-pose method
* Data collection program for segmentation in bin
* Merge pull request `#1793 <https://github.com/start-jsk/jsk_apc/issues/1793>`_ from ban-masa/auto-pressure-calib
  Auto calib pressure threshold
* Use mask image to enhance object recognition result with vgg16 net
* added calib-pressure-threshold
* Prepare for logging
* Use VGG16 net for APC2016 in recognition_in_hand.launch
* Align bounding boxes to robot base frame
* stop using self filter
* Fix position of wait-interpolation-smooth
* Remove :recognize-objects-in-bin in picking-with-sib.l
* Merge pull request `#1784 <https://github.com/start-jsk/jsk_apc/issues/1784>`_ from pazeshun/abort-approach-ik-fail
  Abort picking objects when IK to it fails
* add use-current-pose in ik->bin-entrance
* improve ik->bin-entrance to minimize norm
* Abort picking objects when IK to it fails
* Contributors: Bando Masahiro, Kei Okada, Kentaro Wada, Shingo Kitagawa, Yusuke Niitani, ban-masa, pazeshun

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
