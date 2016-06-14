Run save data script
===========================================


Commands
-------------------

On sheeta (using torso kinect)::
  roslaunch jsk_2016_01_baxter_apc baxter.launch
  roslaunch jsk_2016_01_baxter_apc setup_torso.launch
  roslaunch jsk_2016_01_baxter_apc main.launch json:=$(rospack find jsk_apc2016_common)/json/save_pick_layout_X.json
  roslaunch jsk_2016_01_baxter_apc sib_kinect.launch
  rosparam set /left_hand/target_bin BIN

On a machine where you want to save data
  roslaunch jsk_2016_01_baxter_apc collect_sib_data.launch hand:=LEFT_RIGHT

When changing json
  rosrun jsk_apc2016_common set_rosparams.py $(rospack find jsk_apc2016_common)/json/save_pick_layout_Y.json

Generating json for data collection
  rosrun jsk_apc2016_common collect_sib_data_generate_interface_json.py
