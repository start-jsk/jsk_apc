ARC2017 Stow Task Trial on Real World
=====================================

Stow task trial on real world for ARC2017 can be done on ``baxter@sheeta``.

- Prepare json.
- Set objects in Tote.

.. code-block:: bash

  # Launch nodes to control robot.
  baxter@sheeta $ roslaunch jsk_arc2017_baxter baxter.launch pick:=false

  # Launch nodes in recognition pipeline for stow task.
  baxter@sheeta $ roslaunch jsk_arc2017_baxter setup_for_stow.launch

  # Run task!
  baxter@sheeta $ roslaunch jsk_arc2017_baxter stow.launch json_dir:=$(rospack find jsk_arc2017_common)/data/json/sample_stow_task

