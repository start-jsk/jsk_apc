Run save data script
===========================================


Commands
-------------------

To set up saving nodes::
  roslaunch jsk_2016_01_baxter_apc baxter.launch
  roslaunch jsk_2016_01_baxter_apc setup_torso.launch
  roslaunch jsk_2016_01_baxter_apc setup_softkinetic.launch
  roslaunch jsk_2016_01_baxter_apc segmentation_in_bin.launch json:=$(rospack find jsk_2015_05_baxter_apc)/json/layout_12.json
  roslaunch jsk_2016_01_baxter_apc save_data.launch hand:=right layout_name:=layout_12
  
  
