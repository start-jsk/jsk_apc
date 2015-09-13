^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_2014_baxter_apc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

