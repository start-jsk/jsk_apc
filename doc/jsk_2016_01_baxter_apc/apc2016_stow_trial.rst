APC2016 Stow Task Trial on Real World
=====================================

Stow task trial on real world for APC2016 can be done on ``baxter@sheeta.jsk.imi.i.u-tokyo.ac.jp``.

- Prepare json.
- Setup objects in Kiva.
- Setup objects in Tote.


.. code-block:: bash

  baxter@sheeta $ roscd jsk_apc && git fetch origin
  baxter@sheeta $ git checkout stow-task -b origin/stow-task
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc baxter.launch
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc setup_torso.launch use_stow:=true
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc setup_softkinetic.launch use_stow:=true

  baxter@sheeta $ ssh doura
  baxter@doura $ tmux
  # on a tmux session
  baxter@doura $ sudo -s  # necessary for launch kinect2 with ssh login
  baxter@doura $ roslaunch jsk_2016_01_baxter_apc setup_head.launch
  # detach from the tmux session and logout from doura here

  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc main_stow.launch json:=$(rospack find jsk_apc2016_common)/json/stow_layout_1.json


