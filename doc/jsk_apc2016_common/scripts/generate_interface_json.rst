generate_interface_json.py
==========================

Generate interface json file for Pick & Stow task.

The json id is incremeneted according to the existing json files in
``$(rospack find jsk_apc2016_common)/json``.
Name rule is:

- Json for Pick task: ``pick_layout_[ID].json``
- Json for Stow task: ``stow_layout_[ID].json``


Usage
-----

.. code-block:: bash

  rosrun jsk_apc2016_common generate_interface_json.py

  rosls jsk_apc2016_common/json/  # new json files are generated with next id
