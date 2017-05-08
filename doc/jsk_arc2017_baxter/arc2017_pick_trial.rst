ARC2017 Pick Task Trial on Real World
=====================================

Pick task trial on real world for ARC2017 can be done on ``baxter@sheeta``.

- Prepare json.
- Set objects in Shelf.

.. code-block:: bash

  # Launch nodes to control robot.
  baxter@sheeta $ roslaunch jsk_arc2017_baxter baxter.launch

  # Launch nodes in recognition pipeline for pick task.
  baxter@sheeta $ roslaunch jsk_arc2017_baxter setup_for_pick.launch

  # optional: Check sanity.
  baxter@sheeta $ rosrun jsk_2016_01_baxter_apc check_sanity_setup_for_pick

  # Run task!
  baxter@sheeta $ roslaunch jsk_arc2017_baxter pick.launch json_dir:=$(rospack find jsk_arc2017_common)/data/json/sample_pick_task

