# jsk_apc

<img src="jsk_apc2016_common/resource/icons/icon_white.png" align="right" width="192px" />

[![GitHub version](https://badge.fury.io/gh/start-jsk%2Fjsk_apc.svg)](https://badge.fury.io/gh/start-jsk%2Fjsk_apc)
[![](https://travis-ci.org/start-jsk/jsk_apc.svg?branch=master)](https://travis-ci.org/start-jsk/jsk_apc)
[![Gitter](https://badges.gitter.im/start-jsk/jsk_apc.svg)](https://gitter.im/start-jsk/jsk_apc?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Slack](https://img.shields.io/badge/slack-%23jsk__apc-e100e1.svg)](https://jsk-robotics.slack.com/messages/jsk_apc/)
[![Documentation Status](https://readthedocs.org/projects/jsk-apc/badge/?version=latest)](http://jsk-apc.readthedocs.org/en/latest/?badge=latest)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/jskrobotics/jsk_apc)](https://hub.docker.com/r/jskrobotics/jsk_apc)
**Forum** ([baxter](https://groups.google.com/a/jsk.imi.i.u-tokyo.ac.jp/forum/#!forum/baxter), [apc](https://groups.google.com/a/jsk.imi.i.u-tokyo.ac.jp/forum/#!forum/apc))


**jsk_apc** is a stack of ROS packages for [Amazon Picking Challenge](http://amazonpickingchallenge.org) mainly developed by JSK lab.  
The documentation is available at [here](http://jsk-apc.readthedocs.org).


## Usage

| Competition | Documentation                                                                                             |
|:------------|:----------------------------------------------------------------------------------------------------------|
| APC2015     | See [jsk_2015_05_baxter_apc](http://jsk-apc.readthedocs.org/en/latest/jsk_2015_05_baxter_apc/index.html). |
| APC2016     | See [jsk_2016_01_baxter_apc](http://jsk-apc.readthedocs.org/en/latest/jsk_2016_01_baxter_apc/index.html). |
| ARC2017     | See [jsk_arc2017_baxter](http://jsk-apc.readthedocs.org/en/latest/jsk_arc2017_baxter/index.html).         | 


## Citations

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


## Installation


### Required

1. Install the ROS. [Instructions for ROS indigo on Ubuntu 14.04](http://wiki.ros.org/indigo/Installation/Ubuntu).
2. [Setup your ROS environment](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).
3. Build catkin workspace for [jsk\_apc](https://github.com/start-jsk/jsk_apc):

```sh
$ mkdir -p ~/ros/ws_jsk_apc/src && cd ~/ros/ws_jsk_apc/src
$ wstool init . https://raw.githubusercontent.com/start-jsk/jsk_apc/master/fc.rosinstall.${ROS_DISTRO}
$ cd ..
$ rosdep install -y -r --from-paths .
$ sudo apt-get install python-catkin-tools ros-indigo-jsk-tools
$ catkin build
$ source devel/setup.bash
```

#### Edit `/etc/hosts`:

```
133.11.216.224 baxter 011310P0014.local
```

#### Add below in your `~/.bashrc`:

```
$ rossetmaster baxter
$ rossetip

$ # we recommend below setup (see http://jsk-docs.readthedocs.org/en/latest/jsk_common/doc/jsk_tools/cltools/setup_env_for_ros.html)
$ echo """
rossetip
rosdefault
""" >> ~/.bashrc
$ rossetdefault baxter  # set ROS_MASTER_URI as http://baxter:11311
```


### Optional

#### Setup Arduino and DXHUB

1. To distinguish left DXHUB from right one, follow the instruction [here](http://jsk-apc.readthedocs.io/en/latest/jsk_arc2017_baxter/setup_gripper_v6.html#distinguish-left-dxhub-from-right-one).

2. Create udev rules:
```
$ rosrun jsk_arc2017_baxter create_udev_rules
```
so that Arduinos can appear on `/dev/arduino*` and DXHUBs can appear on `/dev/l_dxhub` and `/dev/r_dxhub`

#### Setup scales

Create udev rules:
```
$ rosrun jsk_arc2017_common create_udev_rules
```
so that scales can appear on `/dev/scale*`

#### Setup SSH

Write below in `~/.ssh/config`:

```
Host baxter
  HostName baxter.jsk.imi.i.u-tokyo.ac.jp
  User ruser  # password: rethink
```

#### Setup UP Board

Inside UP Board...
1. [Install ros-kinetic-ros-base and setup environment](http://wiki.ros.org/kinetic/Installation/Ubuntu).
2. Build catkin workspace for jsk_apc:
```sh
$ source /opt/ros/kinetic/setup.bash
$ mkdir -p ~/ros/kinetic/src && cd ~/ros/kinetic/src
$ wstool init . https://raw.githubusercontent.com/start-jsk/jsk_apc/master/upboard.rosinstall
$ wstool merge -t . https://raw.githubusercontent.com/start-jsk/jsk_apc/master/upboard.rosinstall.kinetic
$ wstool update
$ sudo apt install python-pip
$ rosdep install -y -r --from-paths . --ignore-src
$ sudo apt install python-catkin-tools
$ cd .. && catkin build
$ echo 'source $HOME/ros/kinetic/devel/setup.bash' >> ~/.bashrc
$ echo "rossetip" >> ~/.bashrc
$ echo "rossetmaster baxter" >> ~/.bashrc
$ source ~/.bashrc
```
3. Create udev rules:
```sh
# baxter-c2
$ rosrun jsk_arc2017_baxter create_udev_rules
# baxter-c3
$ rosrun jsk_arc2017_common create_udev_rules
```
4. Create `~/env-loader.sh`:
```sh
#!/bin/bash

. $HOME/ros/kinetic/devel/setup.bash
export ROSCONSOLE_FORMAT='[${severity}] [${time}]: [${node}] [${function}] ${message}'
rossetip
rossetmaster baxter
exec "$@"
```
and `chmod +x ~/env-loader.sh`

5. Setup time synchronization
```sh
sudo apt install ntp
# Set the same configuration as other PCs
sudo vi /etc/ntp.conf
```

From main PC...
1. `ssh -oHostKeyAlgorithms='ssh-rsa' baxter@<UP Board Host Name>.jsk.imi.i.u-tokyo.ac.jp`
2. Add main PC's ssh public key to `~/.ssh/authorized_keys` on UP Board.
