jsk\_2016\_01\_baxter\_apc
==========================


Install
-------


### Required

1. Install the ROS. [Instructions for ROS indigo on Ubuntu 14.04](http://wiki.ros.org/indigo/Installation/Ubuntu).
2. [Setup your ROS environment](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment). **Please make sure that you're using [Shadow Fixed Repository](http://wiki.ros.org/ShadowRepository)**

  ```sh
$ cat /etc/apt/sources.list.d/ros-latest.list
#deb http://packages.ros.org/ros/ubuntu trusty main
deb http://packages.ros.org/ros-shadow-fixed/ubuntu trusty main
$ sudo apt-get update
$ sudo apt-get dist-upgrade
  ```

3. Build catkin workspace for [jsk\_apc](https://github.com/start-jsk/jsk_apc):

```sh
$ mkdir -p ~/ros/ws_jsk_apc/src && cd ~/ros/ws_jsk_apc/src
$ wstool init . https://raw.githubusercontent.com/start-jsk/jsk_apc/master/jsk_2016_01_baxter_apc/rosinstall
$ cd ..
$ rosdep install -y -r --from-paths . --ignore-src
$ catkin build
$ source devel/setup.bash
```

As of 2016/1/27, we're using following version for baxter simulation
```
$ rosrun jsk_2016_01_baxter_apc check_baxter_pkg_version.sh
rosversion baxter_core_msgs     1.2.0
rosversion baxter_description   1.2.0
rosversion baxter_gazebo        1.2.12
rosversion baxter_interface     1.2.0
rosversion baxter_maintenance_msgs 1.2.0
rosversion baxter_sim_controllers  1.2.12
rosversion baxter_sim_hardware  1.2.12
rosversion baxter_sim_io        1.2.12
rosversion baxter_sim_kinematics   1.2.12
rosversion baxter_tools         1.2.0
rosversion baxtereus            1.0.1
```

Usage
-----

### Run Demo on Gazebo Simulator

```sh
$ roslaunch jsk_2016_01_baxter_apc baxter_pick.launch
```


Shared Files
------------

READ/WRITE: https://drive.google.com/drive/u/1/folders/0B9P1L--7Wd2vLXo1TGVYLVh3aE0

Google Drive folder is shared.
There are shared files like log files and datasets.
