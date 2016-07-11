APC2016 Stow Task Trial on Real World
=====================================

Stow task trial on real world for APC2016 can be done on ``baxter@satan`` and ``baxter@eyelash``.

- Prepare json.
- Setup objects in Kiva.
- Setup objects in Tote.


.. code-block:: bash

  # use satan
  baxter@satan $ roscd jsk_apc && git fetch origin
  baxter@satan $ git checkout 1.5.0
  baxter@satan $ roslaunch jsk_2016_01_baxter_apc baxter.launch
  baxter@satan $ roslaunch jsk_2016_01_baxter_apc setup_torso.launch use_stow:=true

  # use eyelash
  baxter@eyelash $ roscd jsk_apc && git fetch origin
  baxter@satan $ git checkout 1.5.0
  baxter@eyelash $ roslaunch jsk_2016_01_baxter_apc setup_astra.launch use_stow:=true

  # use satan
  baxter@satan $ roslaunch jsk_2016_01_baxter_apc main_stow.launch json:=$(rospack find jsk_apc2016_common)/json/stow_layout_1.json


