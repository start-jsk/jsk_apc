^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_arc2017_common
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.1.4 (2017-10-15)
------------------
* Skip ResourceNotFound error to pass build on build.ros.org
* Contributors: Kentaro Wada

4.1.5 (2017-10-23)
------------------
* 4.1.4
* Update CHANGELOG
* Skip ResourceNotFound error to pass build on build.ros.org
* Contributors: Kentaro Wada

4.1.3 (2017-10-12)
------------------

4.1.2 (2017-10-11)
------------------
* Create fcn32s.npz symlink by install_data.py
  In order to avoid error in bloom.
* Contributors: Kentaro Wada

4.1.1 (2017-10-10)
------------------
* Equalize weight_candidates_refiner in jsk_apc to upstream PRs
* Create ItemData by taking photos
* Contributors: Kentaro Wada, Shun Hasegawa

4.1.0 (2017-08-12)
------------------
* Warn about nan value
* Publish nan value when unstable by EkEwIDriver
* use 1B2 instead of K3 for test
* use sample_pick_task json for sample launch
* fix bug in cardboard ids
  1A5 is bigger than 1B2, but 1A5 is never used for cardboard_C and 1B2 for
  cardboard_B
* Contributors: Kentaro Wada, Shingo Kitagawa

4.0.9 (2017-07-29)
------------------
* Fix typo in visualize_object.py
* Contributors: Kentaro Wada

4.0.8 (2017-07-29)
------------------

4.0.7 (2017-07-28)
------------------
* fix visualize_object
* Contributors: Kentaro Wada

4.0.6 (2017-07-28)
------------------

4.0.5 (2017-07-28)
------------------

4.0.4 (2017-07-27)
------------------

4.0.3 (2017-07-27)
------------------
* Set default model
* change suction graspability of avery_binder
* fix nosetest path in jsk_arc2017_common
* Show with red color for unknown  objects in list_label_names
* Add get_known_object_names and its tests
* Fixes for unknown objects
* Use fcn32s.npz in models dir
* Support unknown objects disclosed just before the challenge
  This requires https://github.com/start-jsk/jsk_apc/issues/2414#issuecomment-317935277
* Contributors: Kentaro Wada, Naoya Yamaguchi, Shingo Kitagawa

4.0.2 (2017-07-27)
------------------
* change pinch graspability of white_facecloth
* publish bin_contents in json_saver
* add Content and ContentArray msg
* change reynolds wrap graspability
* Contributors: Naoya Yamaguchi, Shingo Kitagawa

4.0.1 (2017-07-26)
------------------
* Reset pub_imgs
* Classify object in grasped region
* fix apply_mask_to_proba_image.py:32:5: E303 too many blank lines (2) ?
* create apply_mask_to_proba_image.py
* create image_buffer.py
* 148000 iterations with Datasetv3
  ```
  % ./evaluate.py logs/fcn32s_CFG-009_DATASET-v3_OPTIMIZER-Adam_WEIGHT_DECAY-0.0005_LR-1e-05_MAX_ITERATION-150000_VCS-c0bff11_TIME-20170723-112528/models/fcn32s_dataset_v3_cfg009_20170724.npz
  Accuracy: 93.7259
  AccClass: 92.3574
  Mean IoU: 84.6101
  Fwav Acc: 88.3183
  ```
* pinch graspability 2 -> 3
* Contributors: Kei Okada, Kentaro Wada, Naoya Yamaguchi, Yuto Uchimi

4.0.0 (2017-07-24)
------------------
* Fix workorder in shared shelf-bin (B)
* Fix suction for pie plates
* Enhance object location display
* fix graspability of some items
* Fix graspability of speed stick
* FCN32s trained using natural dataset (datasetV3)
* add grasp_style_server.py
* add GetGraspStyle service
* sort work order by graspability
* add list_graspability script
* add func get_object_graspability()
* Fixed training of FCN32s using stacking data augmentation
* No use of ConnectionBasedTransport in WeightCanditatesRefiner
* Fix old timestamp in EkEwIDriver output
* Always subscribe weight scale in weight_candidates_refiner
* Use class segmentation in known objects
  Because we changed the strategy to handle the unknown (newly passed)
  objects.
* Mark ordered objects with red circle in VisualizeJSON
* Fix for pep8
* Update ekew_i_driver.py
* Update ekew_i_driver.py
* Update ekew_i_driver.py
* change topic name
* always publish raw weight value
* Fix typo in weight_candidates_refiner.py
* publish prev_weight_sum for debugging (`#2322 <https://github.com/start-jsk/jsk_apc/issues/2322>`_)
  * publish prev_weight_sum for debugging
  * Update weight_candidates_refiner.py
  * Fix typo
  * Update weight_candidates_refiner.py
* sort work order by object weight, pick lighter one
* add get_object_weights() in jsk_arc2017_common
* move object weight yaml to config dir
* Update README.md
* Disable of downloading old chainer models
* Create dataset V2
* Rename scripts annotate_dataset2d.py, view_dataset2d.py
* Contributors: Kentaro Wada, Naoya Yamaguchi, Shingo Kitagawa, Shun Hasegawa

3.3.0 (2017-07-15)
------------------
* Add script to visualize annotated 2d dataset
* fix E271 multiple spaces after keyword ERROR....
* Detect serial blocked and restart
* Update to support multi shelf bins
* Add README to annotate_2d_dataset
* Rename to annotate_2d_dataset.py
* Publish scenes and view frame of DatasetV3 in ROS
* merge json_generator into one program (`#2270 <https://github.com/start-jsk/jsk_apc/issues/2270>`_)
* Fix for flake8
* Memoize result of visualize_json
* refine weight_candidates_refiner node
* publish -1 when scale is disabled
* remove unused launch
* rename to weight_candidates_refiner node
* add use_topic and input_candidates args
* sub candidates in scale object estimation node
* publish WeightStamped from scale node
* add Weight and WeightStamped msg
* replace bg_label by ignore_labels
* use arc2017 object_segmentation_3d in stow task
* ad ignore_labels in label_to_cpi
* add USE_PCA argment in object_segmentation_3d.launch
* Contributors: Kei Okada, Kentaro Wada, Naoya Yamaguchi, Shingo Kitagawa, Shun Hasegawa

3.2.0 (2017-07-06)
------------------
* add object_classification with FCN launch
* add doc, sample and test for candidates_publisher
* update Label msg API
  follow https://github.com/jsk-ros-pkg/jsk_recognition/pull/2143/commits/109c73fac35f1cdaa13fd31273ca166b2bcbfce9
* add candidates_publisher node
* Create object_segmentation_3d.launch in jsk_arc2017_common
* Semantic segmentation of unknown objects
* Use simlink to scales
* Add udev rule for scale
* Support json with no boxes in visualize_json
* Update doc for visualize_json.py
* Subscribe json_dir input topic in visualize_json
* Publish json_dir in json_saver.py
* Visualize json (item_location/order) for ARC tasks
* copy location and order json in save dir at first
* json_saver supports pick task
* Contributors: Kentaro Wada, Shingo Kitagawa, Shun Hasegawa

3.1.0 (2017-06-30)
------------------
* Add mesh models for 36 objects
* update UpdateJSON and replace SaveJSON by Trigger
* add json_saver.py
* Refactor yaml file format
* Fix format
* Add thread lock to estimate_object_by_scale
* Change init of object estimation to srv
* Add object estimation by scale
* and_scale_rosserial -> ekew_i_driver
* Add object weight data
* remove non-item label in json generator
* update sample_pick_task json
* update pick_json_generator for new pick env
* update work_order_publisher for new shelf
* Place location config files in jsk_arc2017_baxter
* add publish_tote_boxes and interactive tote marker
* Move task config to jsk_arc2017_baxter
* Rename config collect_data -> collect_data_in_shelf
* add files for data collection
* Remove no need merge_depth_images.py
* add depth image merging nodes
* Deploy FCN32s trained on Dataset=v2, config=003
* add shelf_marker for stow_task
* add stow_json_generator and sample_stow_task json
* fix typo in pick_json_generator
* Add python module: get_object_names (`#2132 <https://github.com/start-jsk/jsk_apc/issues/2132>`_)
  * Add python module: get_object_names
  * Fix for flake8
  * Fix typo
* Make label_names.yml as just a name list
* Add log summarization script
* Improve logging in training script
* add easy picking task json files for mayfes demo
* Contributors: Kentaro Wada, Shingo Kitagawa, Shun Hasegawa, Yuto Uchimi

3.0.3 (2017-05-18)
------------------

3.0.2 (2017-05-18)
------------------
* Fix missing build depend on jsk_data
  - because install_data.py is run in Cmake
* Contributors: Kentaro Wada

3.0.1 (2017-05-16)
------------------
* Fix missing dependency on jsk_data
* fix typo in WorkOrderPublisher
* sort cardboard by box size and give ABC name
* Contributors: Kentaro Wada, Shingo Kitagawa

3.0.0 (2017-05-08)
------------------
* Fix style of nodes in roslaunch files
* Add sample for work_order_publisher
* Fix name of sample_set_location_in_rosparam
* Fix for move of data/objects -> config/objects
* Don't use ROS in training script
* add sample launch for set_location_in_rosparam
* print stdout in set_location_in_rosparam
* fix typo in set_location_in_rosparam
* remove unused package and sort alphabetically
* use label_names.yaml instead of objects.txt
* set myself as a author
* update json generator script
* mv pick_work_order_server -> work_order_publisher
* replace publish_shelf_bin_bbox to existing node
* remove unnecessary lines in CMakeLists
* move json -> data/json
* switch cardboard place
  cardboard a: left upper
  cardboard b: left lower
  cardboard c: right
* add abandon items for work_order_server
* fix typo in package.xml in jsk_arc2017_common
* update shelf_bin position config
* set cardboard id as A,B,C in work_order
* add pick_work_order_server test
* fix typo in arc2017 json item_location_file.json
* add myself as a maintainer
* update CMakelists.txt and package.xml for roseus
* add set_location_in_rosparam node
* format bin_name as capital alphabet
* update pick_work_order_server for new json format
* update json generator and sample in correct format
* add example json and box size config
* add pick_work_order_server for arc2017
* introduce new WorkOrder&WorkOrderArray msg
* add sample_pick.json and json generate script
* add setup_for_pick.launch for arc2017
* add shelf_interactive_marker.yaml
* add publish_shelf_bin_bbox for new shelf
* Add python-serial to run_depends
* Fix typo
* Read weight data from AND scale
  - new file:   and_scale_rosserial.py
* Ignore AR20170331
* Update model file with stacking data augmentation
* Add data augmentation method with stacking
* Update api of torchfcn
* Improve imgaug
* Simplify config
* Update data with AR_20170331 dataset
* Add link to wiki
* Neat config & log handling
* Add ROS sample of FCNObjectSegmentation
* Add sample data of JSKV1 dataset
* Fix path of data
* Change path of JSKV1
* Add option to skip dataset with stamp
* Show datetime in annotation
* Improve view_jsk_v1
  - p for back
  - show timestamp
* Training experiments
* Update config
* Check label.npz existence
* Sort dirs for annotation
* Fix locking
* Show stamp_dir
* Lock for parallel annotation
* Augument image using imgaug
* Fix data field name
* 002_fcn32s_dataset_v1.yaml
* Fix for flake8
* Add requirements.txt
* Training script of FCN32s
* Add dataset class for JSKARC2017From16
* Add script to convert JSKAPC2016 to ARC2017
* Split dataset for train and valid
* Remove underscore for consistent names
* Add dataset.py
* Neat visualization of dataset
  - Show size of All and Annotated
  - Show label names
* Script to view dataset before/after annotated
* Update data using Training_items_20170320_fixed.zip
* Save data with compression
* Save label as npz file with compression
* Smaller size of object list
* Annotation script for JSK_V1 dataset
* Add script to list objects
* Visualize object list
* Parse AR_20170224 dataset
* Contributors: Kentaro Wada, Shingo Kitagawa
