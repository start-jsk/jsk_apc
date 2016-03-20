initialize_baxter.py
======================


What is this?
-------------

This node setups below after waiting the ``/clock`` and ``/robot/state`` topics.

  - Enables the robot.
  - Launches ``baxter_interface/joint_trajectory_action_server.py``.
  - Launches ``baxter_interface/head_action_server.py``.


Subscribing Topic
-----------------

None.


Publishing Topic
----------------

None.
