^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_apc2016_common
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
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
