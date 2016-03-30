Demonstrate APC2015 on Simulation
=================================

Real world demonstration for APC2015 can be done on any computers with ROS indigo.

.. code-block:: bash

  roscd jsk_2015_05_baxter_apc && git checkout 0.1.2

  roslaunch jsk_2015_05_baxter_apc baxter_sim.launch kiva:=true
  roslaunch jsk_2015_05_baxter_apc setup.launch
  roslaunch jsk_2015_05_baxter_apc main.launch json:=$(rospack find jsk_apc2015_common)/json/f2.json

  # optional visualization
  rviz -d $(rospack find jsk_2015_05_baxter_apc)/rvizconfig/gazebo_demo.rviz  # visualization for demo


.. image:: _images/apc2015_gazebo_demo.png
   :alt: Amazon Picking Challenge 2015 Gazebo Simulation
   :target: https://www.youtube.com/watch?v=uV6XctamwEA
   :width: 40%
