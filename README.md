jsk\_apc
=======

[![](https://travis-ci.org/start-jsk/jsk_apc.svg)](https://travis-ci.org/start-jsk/jsk_apc)
[![Gitter](https://badges.gitter.im/start-jsk/jsk_apc.svg)](https://gitter.im/start-jsk/jsk_apc?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Install
-------


### Required

1. Install the ROS. [Instructions for ROS indigo on Ubuntu 14.04](http://wiki.ros.org/indigo/Installation/Ubuntu).
2. [Setup your ROS environment](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).
3. Build catkin workspace for [jsk\_apc](https://github.com/start-jsk/jsk_apc):

```sh
$ mkdir -p ~/ros/ws_jsk_apc/src && cd ~/ros/ws_jsk_apc/src
$ wstool init . https://raw.githubusercontent.com/start-jsk/jsk_apc/master/jsk_2015_05_baxter_apc/rosinstall
$ cd ..
$ rosdep install -y -r --from-paths .
$ sudo apt-get install python-catkin-tools ros-indigo-jsk-tools
$ catkin build
$ source devel/setup.bash
```

* Edit `/etc/hosts`:

```
133.11.216.214 baxter 011310P0014.local
```

* Add below in your `~/.bashrc`:
```
$ rossetmaster baxter.jsk.imi.i.u-tokyo.ac.jp
$ rossetip

$ # we recommend below setup (see http://jsk-docs.readthedocs.org/en/latest/jsk_common/doc/jsk_tools/cltools/setup_env_for_ros.html)
$ echo """
rossetip
rosdefault
""" >> ~/.bashrc
$ rossetdefault baxter  # set ROS_MASTER_URI as http://baxter:11311
```


### Optional

* Setup Kinect2: [Instructions at code-iai/iai\_kinect2](https://github.com/code-iai/iai_kinect2#install)
* Setup rosserial + vacuum gripper: Write below in `/etc/udev/rules.d/90-rosserial.rules`:

```
# ATTR{product}=="rosserial"
SUBSYSTEM=="tty", MODE="0666"
```

* Setup SSH: Write below in `~/.ssh/config`:

```
Host baxter
  HostName baxter.jsk.imi.i.u-tokyo.ac.jp
  User ruser  # password: rethink
```


Usage
-----

### Run Demo with Real Robot

```sh
$ roslaunch jsk_2015_05_baxter_apc baxter.launch
$ roslaunch jsk_2015_05_baxter_apc setup.launch
$ roslaunch jsk_2015_05_baxter_apc main.launch json:=`rospack find jsk_2015_05_baxter_apc`/data/apc-a.json
$ roslaunch jsk_2015_05_baxter_apc record.launch  # rosbag record
```


### Run Demo on Gazebo Simulator

```sh
$ roslaunch jsk_2015_05_baxter_apc baxter_sim.launch
$ roslaunch jsk_2015_05_baxter_apc setup.launch
$ roslaunch jsk_2015_05_baxter_apc main.launch json:=`rospack find jsk_apc2015_common`/json/f2.json
```

<a href="https://www.youtube.com/watch?v=uV6XctamwEA">
  <img src="images/apc_gazebo_demo_on_youtube.png" alt="Amazon Picking Challenge 2015 Gazebo Simulation" width="50%" />
</a>


If you have problem...
----------------------

* Run below to synchronize the time with robot. Time synchronization is crucial.:

```
$ sudo ntpdate baxter.jsk.imi.i.u-tokyo.ac.jp
```


Testing
-------

```sh
$ catkin run_tests jsk_2015_05_baxter_apc --no-deps
```
