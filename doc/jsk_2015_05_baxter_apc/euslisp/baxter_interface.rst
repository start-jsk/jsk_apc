Baxter Robot Interface (baxter_interface.l)
===========================================

``baxter-interface.l`` is the implementation of class to activate the
robot in both real and simulated world.


Usage
-----

.. code-block:: bash

  $ roseus $(rospack find jsk_2015_05_baxter_apc)/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l
  euslisp> (jsk_2015_05_baxter_apc::baxter-init)
  euslisp> (send *ri* :angle-vector (send *baxter* :reset-pose))


Methods of ``jsk_2015_05_baxter_apc::baxter-interface``
-------------------------------------------------------

Below are the methods of ``*ri*``.

``:start-grasp``
++++++++++++++++
Start the vacuum gripper.

**Arguments**

  - arm (``:arms``, ``:larm`` or ``:rarm``, default: ``:arms``)

``:stop-grasp``
+++++++++++++++
Stop the vacuum gripper.

**Arguments**

  - arm (``:arms``, ``:larm`` or ``:rarm``, default: ``:arms``)