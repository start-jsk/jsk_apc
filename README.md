jsk_picking_challenge
=====================

[![](https://travis-ci.org/start-jsk/jsk_picking_challenge.svg)](https://travis-ci.org/start-jsk/jsk_picking_challenge)


Install
-------


### Required

1. Install the ROS. [Instructions for ROS indigo on Ubuntu 14.04](http://wiki.ros.org/indigo/Installation/Ubuntu).
2. [Setup your ROS environment](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).
3. Build catkin workspace for [jsk_picking_challenge](https://github.com/wkentaro/2014-semi):

```sh
$ mkdir -p ~/ros/ws_jsk_picking_challenge/src
$ cd ~/ros/ws_jsk_picking_challenge/src
$ wstool init . https://raw.githubusercontent.com/start-jsk/2014-semi/master/jsk_picking_challenge.rosinstall
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
```


### Optional

* Setup Kinect2: [Instructions at code-iai/iai_kinect2](https://github.com/code-iai/iai_kinect2#install)
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

### Run Demo

```sh
$ roslaunch jsk_2015_05_baxter_apc baxter.launch
$ roslaunch jsk_2015_05_baxter_apc setup.launch
$ roslaunch jsk_2015_05_baxter_apc main.launch json:=`rospack find jsk_2015_05_baxter_apc`/data/apc-a.json
$ roslaunch jsk_2015_05_baxter_apc record.launch  # rosbag record
```


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
