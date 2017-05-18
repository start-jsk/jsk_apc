^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_arc2017_baxter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
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
