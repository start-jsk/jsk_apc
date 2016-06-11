Run save data script
===========================================


Commands
-------------------

On sheeta (using torso kinect)::
  roslaunch jsk_2016_01_baxter_apc baxter.launch
  roslaunch jsk_2016_01_baxter_apc setup_torso.launch
  roslaunch jsk_2016_01_baxter_apc main.launch json:=$(rospack find jsk_apc2016_common)/json/save_pick_layout_1.json
  roslaunch jsk_2016_01_baxter_apc sib_kinect.launch
  rosparam set left_target_bin g

On a machine where you want to save data
  roslaunch jsk_2016_01_baxter_apc save_data.launch hand:=LEFT_RIGHT

When changing json
  rosrun jsk_apc2016_common set_rosparams.py $(rospack find jsk_apc2016_common)/json/save_pick_layout_X.json

