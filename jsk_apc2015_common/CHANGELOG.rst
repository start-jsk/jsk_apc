^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_apc2015_common
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
* Update LICENSE
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

4.0.2 (2017-07-27)
------------------

4.0.1 (2017-07-26)
------------------

4.0.0 (2017-07-24)
------------------

3.3.0 (2017-07-15)
------------------
* Fix fcn model_file param name
* Contributors: Kentaro Wada

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

3.0.0 (2017-05-08)
------------------
* Add link to wiki
* use better pod_lowres.stl model
  original repos: https://github.com/mDibyo/apc
* Contributors: Kentaro Wada, Shingo Kitagawa

2.0.0 (2016-10-22)
------------------

1.5.1 (2016-07-15)
------------------
* 1.5.0
* Update CHANGELOG.rst to release 1.5.0
* 1.0.0
* Update CHANGELOG.rst
* Thicker bin name to visualize json
* Thicker target object identification when visualizing json
* 0.8.1
* update CHANGELOG
* 0.8.1
* Update CHANGELOG.rst for 0.8.0
* Contributors: Kei Okada, Kentaro Wada

1.5.0 (2016-07-09)
------------------

1.0.0 (2016-07-08)
------------------
* Thicker bin name to visualize json
* Thicker target object identification when visualizing json
* Contributors: Kentaro Wada

0.8.1 (2016-06-24)
------------------
* Update CHANGELOG.rst for 0.8.0
* Contributors: Kentaro Wada

0.8.0 (2016-05-31)
------------------
* Add vgg16 object_recognition.launch
* Visualize pick json with APC2016 objects
* 2015 launch files do not depend on 2016 config
* Contributors: Kentaro Wada, Yusuke Niitani

0.2.4 (2016-04-15)
------------------
* Add roslint as test_depend
* Contributors: Kentaro Wada

0.2.3 (2016-04-11)
------------------
* Bugfix
  + Install models dir
  + Use python lib not in ros environement
  + Skip bin without contents to visualize
  + No tile shape when no img_num
* Test
  + Add roslint test for python library
  + Test visualize json
* Documentation
  + Doc for python library
  + Move README of jsk_apc2015_common to sphinx
* Cleanup
  + Refactor visualize_json to visualize_bin_contents
* Visualization
  + visualize stow json
* Contributors: Heecheol Kim, Kentaro Wada

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
* Fix for APC2016
  * Dynamic visualization for given number of bin_contents
    Modified:
    - jsk_apc2015_common/src/jsk_apc2015_common/__init_\_.py
* Fix for ONEDO 2015 December
  * Visualize bin_contents and work_order with json
    Modified:
    - jsk_apc2015_common/src/jsk_apc2015_common/__init_\_.py
    Added:
    - jsk_apc2015_common/node_scripts/visualize_json.py
    - jsk_apc2015_common/src/jsk_apc2015_common/util.py
  * Use RosPack for data file
  * Rename function name object_list -> get_object_list
  * [jsk_apc2015_common] Add object image files
    Added:
    - jsk_apc2015_common/models/champion_copper_plus_spark_plug/image.jpg
    - jsk_apc2015_common/models/cheezit_big_original/image.jpg
    - jsk_apc2015_common/models/crayola_64_ct/image.jpg
    - jsk_apc2015_common/models/dr_browns_bottle_brush/image.jpg
    - jsk_apc2015_common/models/elmers_washable_no_run_school_glue/image.jpg
    - jsk_apc2015_common/models/expo_dry_erase_board_eraser/image.jpg
    - jsk_apc2015_common/models/feline_greenies_dental_treats/image.jpg
    - jsk_apc2015_common/models/first_years_take_and_toss_straw_cup/image.jpg
    - jsk_apc2015_common/models/genuine_joe_plastic_stir_sticks/image.jpg
    - jsk_apc2015_common/models/highland_6539_self_stick_notes/image.jpg
    - jsk_apc2015_common/models/kiva_pod/image.jpg
    - jsk_apc2015_common/models/kong_air_dog_squeakair_tennis_ball/image.jpg
    - jsk_apc2015_common/models/kong_duck_dog_toy/image.jpg
    - jsk_apc2015_common/models/kong_sitting_frog_dog_toy/image.jpg
    - jsk_apc2015_common/models/kyjen_squeakin_eggs_plush_puppies/image.jpg
    - jsk_apc2015_common/models/laugh_out_loud_joke_book/image.jpg
    - jsk_apc2015_common/models/mark_twain_huckleberry_finn/image.jpg
    - jsk_apc2015_common/models/mead_index_cards/image.jpg
    - jsk_apc2015_common/models/mommys_helper_outlet_plugs/image.jpg
    - jsk_apc2015_common/models/munchkin_white_hot_duck_bath_toy/image.jpg
    - jsk_apc2015_common/models/oreo_mega_stuf/image.jpg
    - jsk_apc2015_common/models/paper_mate_12_count_mirado_black_warrior/image.jpg
    - jsk_apc2015_common/models/rolodex_jumbo_pencil_cup/image.jpg
    - jsk_apc2015_common/models/safety_works_safety_glasses/image.jpg
    - jsk_apc2015_common/models/sharpie_accent_tank_style_highlighters/image.jpg
    - jsk_apc2015_common/models/stanley_66_052/image.jpg
  * Fix collision of apc_order_bin model
  * Add black stage
  * [jsk_apc2015_common] Add apc order bin
  * Add f2.json
  * [jsk_2015_05_baxter_apc] Light mass param
  * less slippely
  * [jsk_apc2015_common] Fix texture png name for mesh models
  * [jsk_apc2015_common] Lighter objects
  * [jsk_apc2015_common] Test jsk_apc2015_common python package
  * [jsk_apc2015_common] Refactor python package
  * [jsk_apc2015_common] Rename to a.json
  * [jsk_apc2015_common] F2 G1 json
  * jsk_2015_apc_common -> jsk_apc2015_common
  * Add catkin_lint
  * [jsk_2015_apc_common] Add credit for gazebo models
  * [jsk_2015_apc_common] Add gazebo model files
  * [jsk_2015_apc_common] Adjust kiva pod
  * [jsk_2015_apc_common] Update json
  * [jsk_2015_05_baxter_apc] Fix main params
  * [jsk_2015_apc_common] Adjust kiva_pod_interactive_marker
  * [jsk_2015_05_baxter_apc] Remove object_segmentation.launch
  * [jsk_2015_apc_common] Update in_bin_each_object.launch
  * [jsk_2015_apc_common] Update in_bin_atof.launch
  * [jsk_2015_apc_common] Update in_bin_atof.launch
  * [jsk_2015_apc_common] Update in_kiva_pod.launch
  * [jsk_2015_apc_common] Add kiva_pod_filter
  * [jsk_2015_apc_common] Adjust kiva_pod
  * [jsk_2015_apc_common] Add install scripts for data
  * [jsk_2015_apc_common] Rename download script
  * [jsk_2015_apc_common] Add bof object recognition test script
  * [jsk_2015_apc_common] Create trained_data/ and dataset/
  * Add option -O create_mask_applied_dataset.py
  * Add download script and README
  * Add script to create mask applied dataset
  * Add arg in roslaunch files
  * [jsk_2015_apc_common] Keep vision timestamp even if transformed
  * [jsk_2015_apc_common] Increase max_size for object cloud
  * [jsk_2015_apc_common] Fix model path for kiva_pod_filter
  * [jsk_2015_apc_common] gazebo_ros to pass models path to gazebo
  * [jsk_2015_apc_common] kiva_pod -> models/kiva_pod
  * [jsk_2015_apc_common] Move kiva_pod to models dir
  * Revert "[jsk_2015_apc_common] Move kiva_pod model files to urdf/ & meshes/"
    This reverts commit 91a818229d2b6e9faa66912bbbef7370941d30f5.
  * [jsk_2015_apc_common] Move kiva_pod model files to urdf/ & meshes/
  * [jsk_2015_apc_common] keep_organized for each cloud in bin
  * [jsk_2015_apc_common] Change launch syntax arg should be capital
  * [jsk_2015_apc_common] Object clouds in each bin
  * [jsk_2015_apc_common] Add object_segmentation.launch
  * [jsk_2015_apc_common] Segmentation of objects in bin_a
  * [jsk_2015_apc_common] stop creating manager in_bin_atof.launch
  * [jsk_2015_apc_common] Create root topics
  * [jsk_2015_apc_common] Extract pc in each a-f bin
  * [jsk_2015_apc_common] Some ns change of in_kiva_pod.launch
  * [jsk_2015_apc_common] Remap to output
  * [jsk_2015_apc_common] Clip clouds in kiva pod
  * [jsk_2015_apc_common] Add jsk_demo_common as run_depend
  * [jsk_2015_apc_common] Filter kiva pod pointcloud
  * [jsk_2015_apc_common] Add kiva_pod urdf model
  * [jsk_2015_apc_common] Add kiva_pod model
  * [jsk_2015_apc_common] Add python package
  * Move mesh files jsk_2015_05_baxter_apc -> jsk_2015_apc_common
  * Add jsk_2015_apc_common for common programs
* Contributors: Kentaro Wada
