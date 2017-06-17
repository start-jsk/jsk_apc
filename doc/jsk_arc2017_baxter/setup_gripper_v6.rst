Gripper-v6 Setup
================

Adjust gravity compensation
---------------------------

Gripper-v6 is heavy (1.18kg), so we should adjust gravity compensation of Baxter.

For now (2017/6/17), ``roslaunch jsk_arc2017_baxter baxter.launch`` does it by:

.. code-block:: bash

  $ rostopic pub -1 /robot/end_effector/right_gripper/command baxter_core_msgs/EndEffectorCommand '{ id : 131073, command : "configure", args : "{ \"urdf\":{ \"name\": \"right_gripper_mass\", \"link\": [ { \"name\": \"right_gripper_mass\", \"inertial\": { \"mass\": { \"value\": 1.18 }, \"origin\": { \"xyz\": [0.0, 0.0, 0.15] } } } ] }}"}'

If you want to change gripper, you should restore to the original setting by:

.. code-block:: bash

  $ rostopic pub -1 /robot/end_effector/right_gripper/command baxter_core_msgs/EndEffectorCommand '{ id : 131073, command : "configure", args : "{ \"urdf\":{ \"name\": \"right_gripper_mass\", \"link\": [ { \"name\": \"right_gripper_mass\", \"inertial\": { \"mass\": { \"value\": 0 }, \"origin\": { \"xyz\": [0.0, 0.0, 0.0] } } } ] }}"}'

More information about gripper customization of Baxter is on `official page <http://sdk.rethinkrobotics.com/wiki/Gripper_Customization>`_
