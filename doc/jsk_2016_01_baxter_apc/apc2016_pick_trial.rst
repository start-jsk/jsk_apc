APC2016 Pick Task Trial on Real World
=====================================

Pick task trial on real world for APC2016 can be done on ``baxter@sheeta``.

- Prepare json (ex. apc_pick_task_robocup2016.json).
- Setup objects in Kiva.

.. code-block:: bash

  # Launch nodes to control robot.
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc baxter.launch

  # Launch nodes in recognition pipeline for pick task.
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc setup_for_pick.launch

  # optional: Check sanity.
  baxter@sheeta $ rosrun jsk_2016_01_baxter_apc check_sanity_setup_for_pick

  # Launch rviz for GUI to start.
  baxter@sheeta $ rviz -d $(rospack find jsk_2016_01_baxter_apc)/rvizconfig/pick.rviz
  # Or launch rqt_yn_btn.
  baxter@sheeta $ rosrun jsk_rqt_plugins rqt_yn_btn \~:=/rviz/yes_no_button

  # Run task!
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc main.launch json:=$(rospack find jsk_apc2016_common)/json/apc_pick_task_robocup2016.json


Above commands are automated with a single command below:

.. code-block:: bash

   baxter@sheeta $ tmuxinator start apc
