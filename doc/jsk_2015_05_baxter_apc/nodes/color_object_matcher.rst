color_object_matcher.py
=======================


What is this?
-------------

This classifies image for object recognition with color histogram feature.


Subscribing Topic
-----------------

* ``~input`` (``sensor_msgs/Image``)

  Input image.

* ``~input/label`` (``sensor_msgs/Image``)

  Input label image which describes list of region of interest.


Publishing Topic
----------------

* ``~output`` (``jsk_recognition_msgs/ClassificationResult``)

  Classification result of input image for a object set.


Parameters
----------

* ``~queue_size`` (type: ``Int``, default: ``100``)

  Queue size for subscriptions.

* ``~approximate_sync`` (type: ``Bool``, default: ``false``)

  Synchronize policy for ``message_filter``.
