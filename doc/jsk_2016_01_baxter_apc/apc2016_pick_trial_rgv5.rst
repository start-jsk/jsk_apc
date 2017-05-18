APC2016 Pick Task Trial on Real World with Right Gripper-v5
===========================================================

Pick task trial on real world with right gripper-v5 for APC2016 can be done on ``baxter@sheeta``.

- Install right gripper-v5 in Baxter
- Prepare json (ex. test_gripper_v5.json).
- Setup objects in Kiva.

.. code-block:: bash

  # Launch nodes to control robot.
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc baxterrgv5.launch

  # Launch nodes in recognition pipeline for pick task.
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc setup_for_pick.launch

  # optional: Check sanity.
  baxter@sheeta $ rosrun jsk_2016_01_baxter_apc check_sanity_setup_for_pick

  # Run task!
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc main_rgv5.launch json:=$(rospack find jsk_apc2016_common)/json/test_gripper_v5.json
  # even if you pass rviz:=false to main.launch, you need to launch yes_no_button.
  baxter@sheeta $ rosrun jsk_2016_01_baxter_apc yes_no_button


Video
-----

https://drive.google.com/a/jsk.imi.i.u-tokyo.ac.jp/file/d/0BxxBA3J-CunGazhSTVBna2hTMm8/view?usp=sharing
