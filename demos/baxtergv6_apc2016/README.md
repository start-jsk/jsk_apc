# baxtergv6_apc2016


Execute mimic APC2016 tasks with ARC2017 robot system (baxter with gripper-v6).


## Installation

1. [Install jsk_apc](https://github.com/start-jsk/jsk_apc#installation).
2. `rosrun baxtergv6_apc2016 install_data`


## Pick Task Execution

```bash
roslaunch baxtergv6_apc2016 baxter.launch
roslaunch baxtergv6_apc2016 setup_for_pick.launch
roslaunch baxtergv6_apc2016 pick.launch json_dir:=$(rospack find baxtergv6_apc2016)/data/json/pick1
```


## Stow Task Execution

```bash
roslaunch baxtergv6_apc2016 baxter.launch
roslaunch baxtergv6_apc2016 setup_for_stow.launch
roslaunch baxtergv6_apc2016 stow.launch json_dir:=$(rospack find baxtergv6_apc2016)/data/json/stow1
```


## Citation

```
@article{Hasegawa_2019jrm,
  title={A Three-Fingered Hand with a Suction Gripping System for Warehouse Automation},
  author={Shun Hasegawa and Kentaro Wada and Kei Okada and and Masayuki Inaba},
  journal={Journal of Robotics and Mechatronics},
  volume={31},
  number={2},
  pages={289-304},
  year={2019},
  doi={10.20965/jrm.2019.p0289}
}
```
