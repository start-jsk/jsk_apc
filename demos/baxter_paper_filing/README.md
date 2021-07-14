# baxter_paper_filing


Filing a paper by baxterlgv8 (baxter with left gripper-v8).

## Installation

[Install whole jsk_apc](https://github.com/start-jsk/jsk_apc#installation)

## Execution

```bash
roslaunch baxter_paper_filing baxterlgv8.launch
roslaunch baxter_paper_filing setup_for_paper_filing.launch
roseus `rospack find baxter_paper_filing`/euslisp/paper-filing.l

# In Euslisp interpreter
(paper-filing-init :ctype :larm-head-controller :moveit t)
(send *ti* :mainloop)
```

## Demo with [roseus_resume](https://github.com/Affonso-Gui/roseus_resume)

### Installation with roseus_resume

1. [Setup roseus_resume](https://github.com/Affonso-Gui/roseus_resume#setup)
2. Merge jsk_apc workspace into roseus_resume workspace:
   ```bash
   cd ~/roseus_resume_ws/src/
   wstool merge https://raw.githubusercontent.com/start-jsk/jsk_apc/master/fc.rosinstall.${ROS_DISTRO}
   wstool update
   cd ..
   rosdep install -y -r --from-paths .
   catkin build
   source ~/roseus_resume_ws/devel/setup.bash
   ```

### Execution

```bash
roslaunch baxter_paper_filing baxterlgv8.launch ready_for_roseus_resume:=true
roslaunch baxter_paper_filing setup_for_paper_filing.launch
roseus `rospack find baxter_paper_filing`/euslisp/paper-filing-roseus-resume.l

# In Euslisp interpreter
(paper-filing-init :ctype :larm-head-controller :moveit t)
(send *ti* :mainloop)
```

## Sensor Evaluation Experiments

### Installation

See [installation of main demo](#installation)

### Execution


```bash
# Distance Measurement
roseus `rospack find baxter_paper_filing`/euslisp/baxterlgv8-sensor-eval.l
## In Euslisp interpreter
(instance-init)
(dist-eval-init)
(dist-eval)
```

```bash
# Picking Up a Paper
roseus `rospack find baxter_paper_filing`/euslisp/baxterlgv8-sensor-eval.l
## In Euslisp interpreter
(instance-init)
(pick-paper-init)
(pick-paper)
```

```bash
# Picking Up a Paper Box
roseus `rospack find baxter_paper_filing`/euslisp/baxterlgv8-sensor-eval.l
## In Euslisp interpreter
(instance-init)
(pick-paper-init)
(pick-paper-box)
```

```bash
# Picking Up a Paper Box with Conventional Reflection Intensity Sensor
roseus `rospack find baxter_paper_filing`/euslisp/baxterlgv8-sensor-eval.l
## In Euslisp interpreter
(instance-init)
(pick-paper-init)
(pick-paper-box :intensity t)
```

```bash
# Grasping a Paper Box with PR2 tactile sensor
roseus `rospack find baxter_paper_filing`/euslisp/pr2-grasp-paper-box.l
## In Euslisp interpreter
(grasp-paper-box-init :inst t)
(grasp-paper-box)
```

## Video

See [here](https://drive.google.com/file/d/1gMJUKclMXo4LhY9ny9JHIKGhBqRPsTVz/view?usp=sharing)

## Citation

```bib
@ARTICLE{hasegawa2020online,
  author={Hasegawa, Shun and Yamaguchi, Naoya and Okada, Kei and Inaba, Masayuki},
  journal={IEEE Robotics and Automation Letters},
  title={Online Acquisition of Close-Range Proximity Sensor Models for Precise Object Grasping and Verification},
  year={2020},
  volume={5},
  number={4},
  pages={5993-6000},
  doi={10.1109/LRA.2020.3010440}
}
```
