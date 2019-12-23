# vl53l0x_mraa_ros

ROS package for using [VL53L0X](https://www.st.com/en/imaging-and-photonics-solutions/vl53l0x.html) with [libmraa](https://github.com/intel-iot-devkit/mraa).

## Features

- Supports functions of [STMicro API](https://www.st.com/content/st_com/en/products/embedded-software/proximity-sensors-software/stsw-img005.html) by copying code from [Adafruit_VL53L0X](https://github.com/adafruit/Adafruit_VL53L0X)
  - Difference from [ros_vl53l0x](https://github.com/nomumu/ros_vl53l0x)
- ROS messages including whole ranging information from VL53L0X API
- Exports C++ library of class controlling VL53L0X
- Avoids build failure even without libmraa
  - You can use ROS messages only in other PCs which libmraa doesn't support

## Installation

### Install C/C++ Library in libmraa

You can skip this when you only need ROS messages.
```bash
sudo add-apt-repository ppa:mraa/mraa
sudo apt-get update
sudo apt-get install libmraa1 libmraa-dev mraa-tools
```

### Install vl53l0x_mraa_ros

```bash
$ cd ~/catkin_ws/src/
$ git clone https://github.com/pazeshun/vl53l0x_mraa_ros.git
$ cd ..
$ catkin build
$ source devel/setup.bash
```

## Running sample

```bash
$ rosrun vl53l0x_mraa_ros single_vl53l0x_ros
```
