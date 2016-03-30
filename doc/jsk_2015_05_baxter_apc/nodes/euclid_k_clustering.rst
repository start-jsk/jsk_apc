euclid_k_clustering.py
======================


What is this?
-------------

This node dynamically reconfigures the ``~tolerance`` rosparam of
``jsk_pcl_ros/euclid_clustering (jsk_pcl/EuclideanClustering)``
considering the number of objects in the region of interest.


Subscribing Topic
-----------------

* ``~k_cluster`` (``jsk_recognition_msgs/Int32Stamped``)

  Expected number of clusters.

* ``~{node}/cluster_num`` (``jsk_recognition_msgs/Int32Stamped``)

  Actual number of clusters.
  ``{node}`` is the value of rosparam ``~node``. See *Parameters* for detail.


Publishing Topic
----------------

None.


Parameters
----------

* ``~node`` (type: ``String``, required)

  Node name of jsk_pcl_ros/euclid_clustering.

* ``~default_tolerance`` (type: ``Float``, required)

  Default value of ``tolerance``.

* ``~reconfig_eps`` (type: ``Float``, default: ``0.2``)

  Rate of reconfiguration compared to the value at each time.

* ``~reconfig_n_limit`` (type: ``Int``, default ``10``)

  Number of times of reconfiguration.
