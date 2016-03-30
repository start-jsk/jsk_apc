bin_contents.py
===============

What is this?
-------------

Publishes the contents in bins of Kiva Pod whose layout is described in a json file
for Amazon Picking Challenge 2015.


Subscribing Topic
-----------------

None.


Publishing Topic
----------------

* ``~`` (``jsk_2015_05_baxter_apc/BinContentsArray``)

  Bin contents.

* ``~bin_[a-l]_n_object`` (``jsk_recognition_msgs/Int32Stamped``)

  Number of object in each bin.


Parameters
----------

* ``~json`` (type: ``String``, required)

  Path of json file for the challenge.


Example
-------

.. code-block:: bash

  rosrun jsk_2015_05_baxter_apc bin_contents.py _json:=$(rospack find jsk_2015_05_baxter_apc)/json/apc2015_layout_1.json
