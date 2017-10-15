^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_apc2016_common
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.1.4 (2017-10-15)
------------------

4.1.3 (2017-10-12)
------------------

4.1.2 (2017-10-11)
------------------

4.1.1 (2017-10-10)
------------------
* Fix typo: bin/euslint -> scripts/euslint
* Contributors: Kentaro Wada

4.1.0 (2017-08-12)
------------------
* Visualize target0 (index 0 object)
* Contributors: Kentaro Wada

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
* Fixes for unknown objects
* Contributors: Kentaro Wada

4.0.2 (2017-07-27)
------------------

4.0.1 (2017-07-26)
------------------
* Fix line number in euslint
* Install euslint to global bin
* Make it work grasped_region_classifier with resized image
* loosen synchronizer of cpi_decomposer label
* Contributors: Kentaro Wada, Shingo Kitagawa

4.0.0 (2017-07-24)
------------------
* sort target cpi decomposer by cloud_size
* set smaller queue_size for target cpi decomposer
* set smaller queue_size for label cpi decomposer
* resize input of cpi decomposer for largest object selection
* Contributors: Shingo Kitagawa

3.3.0 (2017-07-15)
------------------
* add use_topic and input_candidates args
* replace bg_label by ignore_labels
* ad ignore_labels in label_to_cpi
* Fix fcn model_file param name
* add USE_PCA argment in object_segmentation_3d.launch
* Contributors: Kentaro Wada, Naoya Yamaguchi, Shingo Kitagawa

3.2.0 (2017-07-06)
------------------
* update candidates for segmentation via topic
* Contributors: Shingo Kitagawa

3.1.0 (2017-06-30)
------------------
* update visulization methods
* use fcn in stow task recognition pipeline
* Make label_names.yml as just a name list
* add jsk_gui_msgs to jsk_apc_common
* Pass chainermodel name to correct param
* Contributors: Kei Okada, Kentaro Wada, Shingo Kitagawa, Shun Hasegawa

3.0.3 (2017-05-18)
------------------

3.0.2 (2017-05-18)
------------------
* Put in order tags in CHANGELOG.rst
* Contributors: Kentaro Wada

3.0.1 (2017-05-16)
------------------
* Merge pull request `#2077 <https://github.com/start-jsk/jsk_apc/issues/2077>`_ from knorth55/move-euslint-to-common
  Check euslisp format for jsk_apc2016_common
* fix format for euslint check
* euslint check for samples euslisp file
* mv euslint to jsk_apc2016_common package
* Contributors: Kei Okada, Shingo Kitagawa

3.0.0 (2017-05-08)
------------------
* add arg default for object_segmentation_3d launch
* modify object_segmentation_3d to accept args
* Add json for pick task by baxterrgv5
* Add main launch for baxterrgv5
* Add link to wiki
* Install sample data with a script
* add table plane removal node
* Use compressed images to get them in 30Hz
* Add apc recognition samples with Fetch
* Add script to list object names
* FCN32s-V2: Update fcn32s trained model
  - Trained with dataset v2 (JSK + MIT)
  - 148000 iterations
* Fix 404 of trained data vgg16_rotation_translation_brightness_372000...
* Fix for migrated srv of UpdateTarget
* Support no target in rqt_select_target
* Merge pull request `#1910 <https://github.com/start-jsk/jsk_apc/issues/1910>`_ from start-jsk/mv-srv-to-common
  Move srv to common package to fix dependency graph
* Place yaml file for object data in right place
* Move images under jsk_apc2016_common to use it in launch correctly
* Place node script in right place
* Move srv to common package to fix dependency graph
  - dependency graph should be jsk_2016_01_baxter_apc -> jsk_apc2016_common
* Contributors: Kentaro Wada, Naoya Yamaguchi, Shingo Kitagawa, Shun Hasegawa

2.0.0 (2016-10-22)
------------------
* rqt_select_target use service to update work_order
* reinforce rqt_select_target to show target image
* add rqt_select_target GUI
* rosparam pass work_order bin_contents from json
* Add json for picking demonstration
* Introduce new 3D object segmentation pipeline
  As proposed in https://github.com/start-jsk/jsk_apc/issues/1865
* Add mode to display json with --display
* add publish bin bbox test
* Contributors: Kentaro Wada, Shingo Kitagawa

1.5.1 (2016-07-15)
------------------
* reflected new data & organized all RBO format data & changed name of directory
* ignore sib_rbo_tokyo directory
* flake 8 publish_bin_info
* delete segmentation in bin helper that became unncessary
* delete unnecessary dependency on helper func
* delete all old scripts that are no longer used
* update package.xml maintainers
* Fix CMakeLists.txt to release on apt
* 1.5.0
* Update CHANGELOG.rst to release 1.5.0
* Add apc_stow_task.json for APC2016 real run
* add volume in object_data_2016.yaml
* add in hand recognition for stow task launch
* add stow_layout_2.json
* Merge pull request `#1839 <https://github.com/start-jsk/jsk_apc/issues/1839>`_ from wkentaro/fcn-trained-data
  Add fcn trained data to download
* Fix typo in install_trained_data.py
* Add fcn trained data to download
* Add vgg16 trained_data to download
* 1.0.0
* Update CHANGELOG.rst
* Rename traial json
* Add robocup2016 apc_pick_task.json
* add offset for verifying whether clouds are in bins
* Update chainermodel of VGG16 for rotation/translation/brightness
* difficult layouts list
* manual fix layout
* add three more pick and stow layouts
* change launch to handle debug output
* debug output for fcn
* fcn sib node accepts depth img
* pick task trial
* set parameter used to reject small target mask for fcn
* add second stow and pick layout json
* fix rosparam path for collect_sib_data
* Make water graspability as 4
* skelton for fcn_sib to reject a mask that is too small
* Update vgg16 trained model
* graspability of duct tape updated
* change vgg train data
* Update graspability of gripper2016
* Set respawn=true for vgg16_object_recognition
* fix a bug that messes up pred_label in loop
* sib-fcn publishes label
* expand path with ~ for collect_sib_data
* fcn_node: subtract mean-rgb from input data before doing segmentation
* fcn segmentation in bin node
* gitignore chainermodel:
* Add mode to create mask from BoundingBox not BinInfo
* Merge pull request `#1795 <https://github.com/start-jsk/jsk_apc/issues/1795>`_ from wkentaro/vgg16
  Recognize APC2016 objects with VGG16 network
* Use mask image to enhance object recognition result with vgg16 net
* Add jsk_data to package.xml
* Recognize APC2016 objects with VGG16 network
* 0.8.1
* update CHANGELOG
* 0.8.1
* make object list in alphabetical order
* remove unnecessary log, and make a save-log more informative
* delete unnecessary import
* fix path of install_dataset
* install dataset 2016
* fixed mistake in gitignore
* add update for rbo
* train script for RBO
* add gitignore for jsk_apc2016_common
* collect sib data server
* labelme tool checks if a user has made mistake
* fix: forgotten import publish_target_bin_info
* add default value for rosparam
* print log when target_bin_name is not set
* Fix test for official stow json format
* Visualize official stow json with APC2016 objects
* labelme tool
* rename set_bin_param -> publish_bin_info
* modify publish_bin_info to rospy.Timer
* publish bin bbox node split from publish bin info
* remove header sequence in publish_bin_info
* sort alphabetically in publish_bin_info
* Fix encoding of in bin mask: 8UC1 -> mono8
* raise warning when wrong json is given
* update bin model to measured size
* Merge pull request `#1628 <https://github.com/start-jsk/jsk_apc/issues/1628>`_ from yuyu2172/throttle
  changed log to throttle
* publish_bin_info publishes messages with headers
* fix unsubscribe in rbo_segmentation_in_bin_node
* changed log to throttle
* Merge pull request `#1609 <https://github.com/start-jsk/jsk_apc/issues/1609>`_ from yuyu2172/publish-bin-info-bbox
  publish_bin_info additionally publishes bin's bounding box array
* fix bug: update self.json
* fix line length
* make main loop of rbo_segmentation_in_bin_node simpler
* catch error when rbo raises key error
  Conflicts:
  jsk_apc2016_common/node_scripts/rbo_segmentation_in_bin_node.py
* publish_bin_info now publishes bbox_array
* Merge pull request `#1597 <https://github.com/start-jsk/jsk_apc/issues/1597>`_ from yuyu2172/publish-when-fail
  rbo_segmentation_in_bin_node publishes debug topics even when segmentation fails
* rbo_segmentation_in_bin_node publishes debug topics even when segmentation fails
* read json only when there is update
* publish_bin_info publishes bin_info of the current json rosparam
* visualize posterior overlaid with color
* Update CHANGELOG.rst for 0.8.0
* Contributors: Kei Okada, Kentaro Wada, Shingo Kitagawa, Yusuke Niitani

1.5.0 (2016-07-09)
------------------
* Add apc_stow_task.json for APC2016 real run
* add volume in object_data_2016.yaml
* add in hand recognition for stow task launch
* add stow_layout_2.json
* Merge pull request `#1839 <https://github.com/start-jsk/jsk_apc/issues/1839>`_ from wkentaro/fcn-trained-data
  Add fcn trained data to download
* Fix typo in install_trained_data.py
* Add fcn trained data to download
* Add vgg16 trained_data to download
* Contributors: Kentaro Wada, Shingo Kitagawa

1.0.0 (2016-07-08)
------------------
* Rename traial json
* Add robocup2016 apc_pick_task.json
* add offset for verifying whether clouds are in bins
* Update chainermodel of VGG16 for rotation/translation/brightness
* difficult layouts list
* manual fix layout
* add three more pick and stow layouts
* change launch to handle debug output
* debug output for fcn
* fcn sib node accepts depth img
* pick task trial
* set parameter used to reject small target mask for fcn
* add second stow and pick layout json
* fix rosparam path for collect_sib_data
* Make water graspability as 4
* skelton for fcn_sib to reject a mask that is too small
* Update vgg16 trained model
* graspability of duct tape updated
* change vgg train data
* Update graspability of gripper2016
* Set respawn=true for vgg16_object_recognition
* fix a bug that messes up pred_label in loop
* sib-fcn publishes label
* expand path with ~ for collect_sib_data
* fcn_node: subtract mean-rgb from input data before doing segmentation
* fcn segmentation in bin node
* gitignore chainermodel:
* Add mode to create mask from BoundingBox not BinInfo
* Merge pull request `#1795 <https://github.com/start-jsk/jsk_apc/issues/1795>`_ from wkentaro/vgg16
  Recognize APC2016 objects with VGG16 network
* Use mask image to enhance object recognition result with vgg16 net
* Add jsk_data to package.xml
* Recognize APC2016 objects with VGG16 network
* remove unnecessary log, and make a save-log more informative
* delete unnecessary import
* Contributors: Kentaro Wada, Yusuke Niitani

0.8.1 (2016-06-24)
------------------
* make object list in alphabetical order
* fix path of install_dataset
* install dataset 2016
* fixed mistake in gitignore
* add update for rbo
* train script for RBO
* add gitignore for jsk_apc2016_common
* collect sib data server
* labelme tool checks if a user has made mistake
* fix: forgotten import publish_target_bin_info
* add default value for rosparam
* print log when target_bin_name is not set
* Fix test for official stow json format
* Visualize official stow json with APC2016 objects
* labelme tool
* rename set_bin_param -> publish_bin_info
* modify publish_bin_info to rospy.Timer
* publish bin bbox node split from publish bin info
* remove header sequence in publish_bin_info
* sort alphabetically in publish_bin_info
* Fix encoding of in bin mask: 8UC1 -> mono8
* raise warning when wrong json is given
* update bin model to measured size
* Merge pull request `#1628 <https://github.com/start-jsk/jsk_apc/issues/1628>`_ from yuyu2172/throttle
  changed log to throttle
* publish_bin_info publishes messages with headers
* fix unsubscribe in rbo_segmentation_in_bin_node
* changed log to throttle
* Merge pull request `#1609 <https://github.com/start-jsk/jsk_apc/issues/1609>`_ from yuyu2172/publish-bin-info-bbox
  publish_bin_info additionally publishes bin's bounding box array
* fix bug: update self.json
* fix line length
* make main loop of rbo_segmentation_in_bin_node simpler
* catch error when rbo raises key error
  Conflicts:
  jsk_apc2016_common/node_scripts/rbo_segmentation_in_bin_node.py
* publish_bin_info now publishes bbox_array
* Merge pull request `#1597 <https://github.com/start-jsk/jsk_apc/issues/1597>`_ from yuyu2172/publish-when-fail
  rbo_segmentation_in_bin_node publishes debug topics even when segmentation fails
* rbo_segmentation_in_bin_node publishes debug topics even when segmentation fails
* read json only when there is update
* publish_bin_info publishes bin_info of the current json rosparam
* visualize posterior overlaid with color
* Update CHANGELOG.rst for 0.8.0
* Contributors: Kentaro Wada, Shingo Kitagawa, Yusuke Niitani

0.8.0 (2016-05-31)
------------------
* Fix using float object not rospy.Rate in publish_target_bin_info.py
* Visualize segementation result in bin
* Merge pull request `#1569 <https://github.com/start-jsk/jsk_apc/issues/1569>`_ from yuyu2172/image-resize
  resize rgb image from softkinetics to the size of depth
* make tf_bbox compatiable with binning_x and binning_y
* deleted compressed target mask
* Use timer to publish target bin info periodically
* segmentation_in_bin nodes continue to run when bin_info_array is not published
* add get_object_data graspability test checking range in [1, 4]
* get_object_data test added gripper2016 key existance
* graspability of gripper2015 updated: rolodex_jumbo_pencil_cup
* Add graspability of new gripper
* get_object_data test added gripper2015 key existance
* object_data_2016 yaml style fixed
* Merge pull request `#1542 <https://github.com/start-jsk/jsk_apc/issues/1542>`_ from wkentaro/visualize-2016
  [jsk_apc2016_common] Visualize pick json with APC2016 objects
* Add cmake dependency on jsk_apc2016_common
* move get_work_order and get_bin_contents func to jsk_apc2016_common
* Visualize pick json with APC2016 objects
* Add object images for apc2016
* add header to sync msg
* tf_bbox_to_mask produces warning message when posiiton of an arm is incorrect
* [jsk_2016_01_baxter_apc | jsk_apc2016_common] CMakeLists syntax fixed
* rbo_segmentation_in_bin_node returns nothing when it fails to predict anyhting
* update comment out in get_object_data
* Revert "[jsk_apc2016_common] publish_bin_tf now uses tf2_ros static_tf_publisher"
* publish_bin_tf now uses tf2_ros static_tf_publisher
* compress rbo mask image to point cloud size
* removed patch on rbo_sib that fixes time stamp to now
* fixed handling of empty target_bin_name rosparam
* publishes posterior images as topic
* cloud_to_spatial_features deal with the case when tf frames of bin are not published
* fixed tf_bbox_to_mask's callback queue_size
* rbo_segmentation_in_bin now takes synchronized messages as input
* topic synchronizer converts 4 images to one msg
* fix publish target_bin_info to sleep a little in each cycle
* add segmentation_in_bin node which is much thinner than previous one
* add sib_spatial_preprocessing node
* move tf_bbox_to_mask to jsk_apc2016_common
* Add officially distributed json files
* Fix for pep8
* Feature to generate identical interface json file
* Fix style of code 'generate_interface_json.py'
* Enhance the interface of arguments for validating script
* Add scripts for interface json from APC2016 official
* alphabetic sorted object_data_2016
* test get_object_data for apc2016
* modify get_object_data func to load apc2016 objects list
* apc2016 object name fixed
* add publish target_bin
* split publish tf and publish bin info
* fixed quaternion of bin param
* add header to BinInfo so that frame of bin is included
* publish bin's tf
* publish_bin_info method became more modular
* add segmenter setup bash script
* rbo_segmentation submodule update
* deleted confusing setters
* fixed value for undetermined pixel for depth
* ignore trained segmenter
* scaled masked image pixel values
* changed name of topic_synchronizer
* add cpp message synchronizer
* unzoom returned prediction
* use rospy debug tools
  print -> rospy.loginfo
  error IO -> rospy.logerr
* 2015 launch files do not depend on 2016 config
* add rbo_segmentation_in_bin that connects different codes
* make .yaml compatiable with 2015 code
* add a node that publishes BinInfoArray from json
* add helper functions for segmentation_in_bin
* add BinData which adds extra information to BinInfo
* add tests for spatial feature extractions
* add spatial feature extractions
* moved mask_bin to rbo_preprocessing
* add BinInfo.msg and BinInfoArray.msg
* add functions that generate mask image of the target bin
* add training script for rbo's segmentation
* update rbo_segmentation's submodule
* Add condition for not initialized submodule
* add rbo's code as submodule
* Exclude rbo_segmentation code from roslint_python
* Contributors: Kentaro Wada, Shingo Kitagawa, Yusuke Niitani, pazeshun

0.2.4 (2016-04-15)
------------------

0.2.3 (2016-04-11)
------------------
* Data
  + add apc2016 object_data
* Test
  + Add roslint_test for python library
  + Add test for python library
* Data
  + Doc for python lib
* Visualization
  + Visualize json for stow task
  + visualize stow json
* Contributors: Heecheol Kim, Kentaro Wada, Shingo Kitagawa

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
* Initialize common package for APC2016
  * Fix version number of jsk_apc2016_common
  * Add object data for APC2016
* Contributors: Kentaro Wada
