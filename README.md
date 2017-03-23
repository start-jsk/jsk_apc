jsk\_apc
=======

<img src="jsk_apc2016_common/resource/icons/icon_white.png" align="right" width="192px" />

[![GitHub version](https://badge.fury.io/gh/start-jsk%2Fjsk_apc.svg)](https://badge.fury.io/gh/start-jsk%2Fjsk_apc)
[![](https://travis-ci.org/start-jsk/jsk_apc.svg)](https://travis-ci.org/start-jsk/jsk_apc)
[![Gitter](https://badges.gitter.im/start-jsk/jsk_apc.svg)](https://gitter.im/start-jsk/jsk_apc?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Slack](https://img.shields.io/badge/slack-%23jsk__apc-e100e1.svg)](https://jsk-robotics.slack.com/messages/jsk_apc/)
[![Documentation Status](https://readthedocs.org/projects/jsk-apc/badge/?version=latest)](http://jsk-apc.readthedocs.org/en/latest/?badge=latest)


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

See [jsk_2015_05_baxter_apc](http://jsk-apc.readthedocs.org/en/latest/jsk_2015_05_baxter_apc/index.html).


Install
-------


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

**Setup Kinect2**

Please follow [Instructions at code-iai/iai\_kinect2](https://github.com/code-iai/iai_kinect2#install),
however, maybe you have error with the master branch. In that case, please use
[this rosinstall](https://github.com/start-jsk/jsk_apc/blob/master/kinect2.rosinstall).

**Setup rosserial + vacuum gripper**

Write below in `/etc/udev/rules.d/90-rosserial.rules`:

```
# ATTR{product}=="rosserial"
SUBSYSTEM=="tty", MODE="0666"
```

**Setup SSH**

Write below in `~/.ssh/config`:

```
Host baxter
  HostName baxter.jsk.imi.i.u-tokyo.ac.jp
  User ruser  # password: rethink
```

**Setup Softkinetic Camera**

See [here](http://jsk-recognition.readthedocs.org/en/latest/install_softkinetic_camera.html) for almost all install process,
but for installing ROS package please do like below:

```bash
git clone https://github.com/knorth55/softkinetic.git -b jsk_apc
cd softkinetic/softkinetic_camera
catkin bt -iv
```
