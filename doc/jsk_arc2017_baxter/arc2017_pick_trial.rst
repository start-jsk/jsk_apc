ARC2017 Pick Task Trial on Real World
=====================================

Pick task trial on real world for ARC2017 can be done on ``baxter@baxter-c1``.

- Prepare json.
- Set objects in Shelf.

.. code-block:: bash

  # Launch nodes to control robot.
  baxter@baxter-c1 $ roslaunch jsk_arc2017_baxter baxter.launch

  # Launch nodes in recognition pipeline for pick task.
  baxter@baxter-c1 $ roslaunch jsk_arc2017_baxter setup_for_pick.launch

  # optional: Check sanity.
  baxter@baxter-c1 $ rosrun jsk_2016_01_baxter_apc check_sanity_setup_for_pick

  # Run task!
  baxter@baxter-c1 $ roslaunch jsk_arc2017_baxter pick.launch json_dir:=$(rospack find jsk_arc2017_common)/data/json/sample_pick_task

With Environment Imitating ARC2017 Pick Competition
---------------------------------------------------

Preparation
^^^^^^^^^^^

.. code-block:: bash

  baxter@baxter-c1 $ rosrun jsk_arc2017_common install_pick_re-experiment

Execution
^^^^^^^^^

.. code-block:: bash

  baxter@baxter-c1 $ roslaunch jsk_arc2017_baxter baxter.launch
  baxter@baxter-c1 $ roslaunch jsk_arc2017_baxter setup_for_pick.launch
  baxter@baxter-c1 $ roslaunch jsk_arc2017_baxter pick.launch json_dir:=$HOME/data/arc2017/system_inputs_jsons/pick_re-experiment/json

