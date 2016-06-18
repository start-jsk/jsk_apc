APC2016 Pick Task Trial on Real World
=====================================

Pick task trial on real world for APC2016 can be done on ``baxter@satan`` and ``baxter@eyelash``.

- Prepare json.
- Setup objects in Kiva.

.. code-block:: bash

  # use satan
  baxter@satan $ roslaunch jsk_2016_01_baxter_apc baxter.launch
  baxter@satan $ roslaunch jsk_2016_01_baxter_apc setup_torso.launch

  # use eyelash
  baxter@eyelash $ roslaunch jsk_2016_01_baxter_apc setup_astra.launch

  # use satan again
  baxter@satan $ roslaunch jsk_2016_01_baxter_apc main.launch json:=$(rospack find jsk_apc2016_common)/json/pick_layout_1.json

