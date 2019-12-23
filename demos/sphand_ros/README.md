# sphand_ros

Packages which provide ROS tools for the Suction Pinching Hand after Ver.7.0.

## Installation

### Install Ubuntu and ROS to UP Board

#### Ubuntu 14.04 and ROS Indigo

Please follow <https://01.org/developerjourney/installing-ubutnu-1404-lts-intel-realsense-robotic-development-kit>.

(2019/10/10) The above page is lost, so see [the archived page](https://web.archive.org/web/20190117004901/https://01.org/developerjourney/installing-ubutnu-1404-lts-intel-realsense-robotic-development-kit).

- Caution

  Please do `sudo apt-get -y autoremove --purge 'linux-.*generic'` again after rebooting for installing `linux-upboard`.

#### Ubuntu 16.04 and ROS Kinetic

1. Follow "Install Ubuntu for UP, UP2, UP Core and UP Core Plus" and "Install Ubuntu kernel 4.15.0 for UP from PPA on Ubuntu 16.04" in <https://wiki.up-community.org/Ubuntu>

2. Follow <http://wiki.ros.org/kinetic/Installation/Ubuntu> to install `ros-kinetic-desktop-full`.

### Install Astra Camera

Please follow [jsk_recognition documentation](https://jsk-recognition.readthedocs.io/en/latest/install_astra_camera.html).

### Install C/C++ Library in libmraa

```bash
sudo add-apt-repository ppa:mraa/mraa
sudo apt-get update
sudo apt-get install libmraa1 libmraa-dev mraa-tools
```
Details are on [GitHub page](https://github.com/intel-iot-devkit/mraa).

### Install sphand_ros and its dependencies

```bash
mkdir -p ~/apc_ws/src
cd ~/apc_ws/src
git clone https://github.com/pazeshun/sphand_ros.git
ln -s sphand_ros/fc.rosinstall .rosinstall
wstool update
cd JSK_APC_WS
# `ln -s jsk_apc/.travis.rosinstall .rosinstall` is better, but .travis.rosinstall.$ROS_DISTRO is also needed
cp jsk_apc/.travis.rosinstall .rosinstall
wstool merge jsk_apc/.travis.rosinstall.$ROS_DISTRO
wstool update
cd ~/apc_ws/src
rosdep install -y -r --from-paths . --ignore-src
sudo apt-get install python-catkin-tools
cd ..
catkin build
source ~/apc_ws/devel/setup.bash
```

### Setup I<sup>2</sup>C, SPI and DXHUB

#### Add permission to device files

```bash
rosrun sphand_driver create_udev_rules
sudo reboot
```

#### Enable second SPI port

Your system may not have the device file of second SPI port `/dev/spidev2.1`:
```bash
$ ls -l /dev/spidev2.1
ls: cannot access '/dev/spidev2.1': No such file or directory
```
In that case, you have to create it by adding an extra kernel boot parameter.
1. Open `/etc/default/grub` in some editor (e.g., `$ sudo vi /etc/default/grub`)
2. Find the line starting with `GRUB_CMDLINE_LINUX_DEFAULT` and append `up_board.spidev1=Y` to its end. For example:
   ```
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash up_board.spidev1=Y"
   ```
   Save and close the file
3. Apply that change to the booting system by `$ sudo update-grub`
4. `$ sudo reboot`
5. Check if `/dev/spidev2.1` exists:
   ```bash
   $ ls -l /dev/spidev2.1
   crw-rw-rw- 1 root root 153, 1 10月 17 16:16 /dev/spidev2.1
   ```
See <https://github.com/pazeshun/sphand_ros/issues/11> for more details.

### Setup dynamixel motors

1. Unite baudrate to 57600 by `rosrun dynamixel_driver set_servo_config.py`
2. Set unique ID by `rosrun dynamixel_driver change_id.py`
3. Disable Overload Error in Alarm LED and Alarm Shutdown of finger\_tendon\_winder by following
```
$ sudo apt-get install ipython
$ ipython

In [1]: import roslib

In [2]: roslib.load_manifest('dynamixel_driver')

In [3]: from dynamixel_driver import dynamixel_io

In [4]: dxl_io = dynamixel_io.DynamixelIO("/dev/dxhub", 57600)

In [5]: dxl_io.write(3, 17, (4,))  # Please check Alarm LED address is 17

In [6]: dxl_io.write(3, 18, (4,))  # Please check Alarm Shutdown address is 18
```

### Setup time synchronization

```bash
sudo apt-get install ntp
# Set the same configuration as other PCs. Don't forget to set disable monitor for security
sudo vi /etc/ntp.conf
```

### Setup auto launching

**CAUTION**
The following procedure assumes that the output of `i2cdetect` includes string "no" when I<sup>2</sup>C interface is correctly recognized.
Otherwise, UP Board will reboot infinitely.
Check `i2cdetect` before running this script.

```bash
$ sudo apt-get install supervisor
# i2cdetect, tmux, rossetip & rossetmaster are required for auto-launching scripts
$ sudo apt-get install i2c-tools tmux ros-$ROS_DISTRO-jsk-tools
$ rosrun sphand_driver create_supervisor_conf

This script generates supervisor config for auto launching
in /etc/supervisor/conf.d

Which user do you want to launch sphand_driver? baxter
Use baxter as user name in configs
$ sudo service supervisor restart
```

\* Intended output of `i2cdetect`:
```bash
$ i2cdetect -F 1
Functionalities implemented by /dev/i2c-1:
I2C                              yes
SMBus Quick Command              no
SMBus Send Byte                  yes
SMBus Receive Byte               yes
SMBus Write Byte                 yes
SMBus Read Byte                  yes
SMBus Write Word                 yes
SMBus Read Word                  yes
SMBus Process Call               no
SMBus Block Write                no  # yes in 16.04
SMBus Block Read                 no  # yes in 16.04
SMBus Block Process Call         no
SMBus PEC                        no
I2C Block Write                  yes
I2C Block Read                   yes
```
In new UP Board, `/dev/i2c-5` is used, but `i2cdetect -F 1` is checked in auto-starting script for backward compatibility.
See <https://github.com/pazeshun/sphand_ros/issues/12> for background.

## Usage

### How to restart roslaunch

```bash
sudo service supervisor restart  # Or {sudo supervisorctl restart}
```

### How to access roslaunch terminal

```bash
# Be careful to use this, because this session is killed when logger daemon is killed or roslaunch daemon is restarted
tmux attach -t gripper
```

### How to see stdout/stderr from roslaunch dropped from tmux

```bash
cat /var/log/supervisor/LaunchLogger.log
```

### How to see other logs

```bash
cat /var/log/supervisor/CheckI2cdetect.log  # Log of I2C interface recognition test
cat /var/log/supervisor/LaunchLeftGripper.log  # Log of preprocess for roslaunch (e.g., network checking)
```
