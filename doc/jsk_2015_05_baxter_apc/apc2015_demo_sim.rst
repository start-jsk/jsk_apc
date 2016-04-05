Demonstrate APC2015 on Simulation
=================================

Real world demonstration for APC2015 can be done on any computers with ROS indigo.


Installation
------------

- Install the ROS: (`Instructions for ROS indigo on Ubuntu 14.04 <http://wiki.ros.org/indigo/Installation/Ubuntu>`_).
- Setup catkin workspace as below:

.. code-block:: bash

  # setup catkin
  mkdir -p ~/ros/jsk_apc2015_sim && cd ~/ros/jsk_apc2015_sim
  catkin init
  # setup repos
  cd ~/ros/jsk_apc2015_sim/src
  wstool init
  wstool merge https://raw.githubusercontent.com/start-jsk/jsk_apc/0.1.3/jsk_2015_05_baxter_apc/simulation.rosinstall
  wstool update -j8
  # install depends
  rosdep install --from-path . -r -y
  # build repos
  catkin build -iv -j8


Demo
----

.. code-block:: bash

  roslaunch jsk_2015_05_baxter_apc baxter_sim.launch
  roslaunch jsk_2015_05_baxter_apc setup.launch
  roslaunch jsk_2015_05_baxter_apc main.launch json:=$(rospack find jsk_apc2015_common)/json/f2.json

  # optional visualization
  rviz -d $(rospack find jsk_2015_05_baxter_apc)/rvizconfig/gazebo_demo.rviz


.. image:: _images/apc2015_gazebo_demo.png
   :alt: Amazon Picking Challenge 2015 Gazebo Simulation
   :target: https://www.youtube.com/watch?v=uV6XctamwEA
   :width: 40%
