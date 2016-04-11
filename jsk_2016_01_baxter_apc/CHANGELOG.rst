^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jsk_2016_01_baxter_apc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
