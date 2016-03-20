boost_object_recognition.py
===========================


What is this?
-------------

This node considers each classifier's weight for the object recognition result.


Subscribing Topic
-----------------

* ``~input/bof`` (``jsk_recognition_msgs/ClassificationResult``)

  Result of classification with Bag of Features.

* ``~input/ch`` (``jsk_recognition_msgs/ClassificationResult``)

  Result of classification with Color Histogram.


Publishing Topic
----------------

* ``~output`` (``jsk_recognition_msgs/ClassificationResult``)

  Result of boosting.


Parameters
----------

* ``~weight`` (type: ``String``, required)

  Path to yaml file for boosting weight.

* ``~queue_size`` (type: ``Int``, default: ``100``)

  Queue size for subscriptions.

* ``~approximate_sync`` (type: ``Bool``, default: ``false``)

  Synchronize policy for ``message_filter``.
