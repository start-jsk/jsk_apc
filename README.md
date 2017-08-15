jsk\_apc
=======

<img src="jsk_apc2016_common/resource/icons/icon_white.png" align="right" width="192px" />

[![GitHub version](https://badge.fury.io/gh/start-jsk%2Fjsk_apc.svg)](https://badge.fury.io/gh/start-jsk%2Fjsk_apc)
[![](https://travis-ci.org/start-jsk/jsk_apc.svg?branch=master)](https://travis-ci.org/start-jsk/jsk_apc)
[![Gitter](https://badges.gitter.im/start-jsk/jsk_apc.svg)](https://gitter.im/start-jsk/jsk_apc?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Slack](https://img.shields.io/badge/slack-%23jsk__apc-e100e1.svg)](https://jsk-robotics.slack.com/messages/jsk_apc/)
[![Documentation Status](https://readthedocs.org/projects/jsk-apc/badge/?version=latest)](http://jsk-apc.readthedocs.org/en/latest/?badge=latest)
[![Docker Build Status](https://img.shields.io/docker/build/wkentaro/jsk_apc.svg)](https://hub.docker.com/r/wkentaro/jsk_apc)


**jsk_apc** is a stack of ROS packages for [Amazon Picking Challenge](http://amazonpickingchallenge.org) mainly developed by JSK lab.  
The documentation is available at [here](http://jsk-apc.readthedocs.org).


Build Status
------------

| Package         | Indigo (Trusty)                                                                                                                                                                          |
|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| jsk_apc (i386)  | [![Build Status](http://build.ros.org/job/Ibin_uT32__jsk_apc__ubuntu_trusty_i386__binary/badge/icon)](http://build.ros.org/job/Ibin_uT32__jsk_apc__ubuntu_trusty_i386__binary)           |
| jsk_apc (amd64) | [![Build Status](http://build.ros.org/job/Ibin_uT64__jsk_apc__ubuntu_trusty_amd64__binary/badge/icon)](http://build.ros.org/job/Ibin_uT64__jsk_apc__ubuntu_trusty_amd64__binary)         |


Usage
-----

| Competition | Documentation                                                                                             |
|:------------|:----------------------------------------------------------------------------------------------------------|
| APC2015     | See [jsk_2015_05_baxter_apc](http://jsk-apc.readthedocs.org/en/latest/jsk_2015_05_baxter_apc/index.html). |
| APC2016     | See [jsk_2016_01_baxter_apc](http://jsk-apc.readthedocs.org/en/latest/jsk_2016_01_baxter_apc/index.html). |
| ARC2017     | See [jsk_arc2017_baxter](http://jsk-apc.readthedocs.org/en/latest/jsk_arc2017_baxter/index.html).         | 


Citations
---------

```
# Our system at APC2015
@article{wada2017pick,
  title={Pick-and-verify: verification-based highly reliable picking system for various target objects in clutter},
  author={Wada, Kentaro and Sugiura, Makoto and Yanokura, Iori and Inagaki, Yuto and Okada, Kei and Inaba, Masayuki},
  journal={Advanced Robotics},
  volume={31},
  number={6},
  pages={311--321},
  year={2017},
  publisher={Taylor \& Francis}
}
```


Installation
------------


### Required

1. Install the ROS. [Instructions for ROS indigo on Ubuntu 14.04](http://wiki.ros.org/indigo/Installation/Ubuntu).
2. [Setup your ROS environment](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).
3. Build catkin workspace for [jsk\_apc](https://github.com/start-jsk/jsk_apc):

```sh
$ mkdir -p ~/ros/ws_jsk_apc/src && cd ~/ros/ws_jsk_apc/src
$ wstool init . https://raw.githubusercontent.com/start-jsk/jsk_apc/master/fc.rosinstall
$ cd ..
$ rosdep install -y -r --from-paths .
$ sudo apt-get install python-catkin-tools ros-indigo-jsk-tools
$ catkin build
$ source devel/setup.bash
```

* Edit `/etc/hosts`:

```
133.11.216.224 baxter 011310P0014.local
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

**Setup Arduino and DXHUB**

Create udev rules:
```
$ rosrun jsk_arc2017_baxter create_udev_rules
```
so that Arduinos can appear on `/dev/arduino*` and DXHUB can appear on `/dev/r_dxhub`

**Setup scales**

Create udev rules:
```
$ rosrun jsk_arc2017_common create_udev_rules
```
so that scales can appear on `/dev/scale*`

**Setup SSH**

Write below in `~/.ssh/config`:

```
Host baxter
  HostName baxter.jsk.imi.i.u-tokyo.ac.jp
  User ruser  # password: rethink
```
