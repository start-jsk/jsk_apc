APC2016 Pick Task Trial on Real World
=====================================

Pick task trial on real world for APC2016 can be done on ``baxter@satan`` and ``baxter@eyelash``.

- Prepare json.
- Setup objects in Kiva.

.. code-block:: bash

  # use sheeta
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc baxter.launch
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc setup_torso.launch

  # use eyelash
  baxter@eyelash $ roslaunch jsk_2016_01_baxter_apc setup_astra.launch

  # use boa or a computer with a good GPU
  baxter@boa $ roslaunch jsk_2016_01_baxter_apc fcn_segmentation_in_bin.launch

  # use sheeta again
  baxter@sheeta $ roslaunch jsk_2016_01_baxter_apc main.launch json:=$(rospack find jsk_apc2016_common)/json/pick_layout_1.json

