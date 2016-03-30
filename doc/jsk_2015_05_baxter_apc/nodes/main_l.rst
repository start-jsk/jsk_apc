main.l
======


What is this?
-------------

Main program to activate robot which subscribes recognition result and does manipulation.


Subscribing/Publishing Topic
----------------------------

Subscriptions and publications are done in below see them:

- ``pr2eus/robot-interface.l``
- ``baxtereus/baxter-interface.l``
- ``jsk_2015_05_baxter_apc/euslisp/jsk_2015_05_baxter_apc/baxter-interface.l``.


Parameters
----------

* ``~[left,right]_hand/state`` (type: ``String``, *Do not set manually*)
* ``~[left,right]_hand/target_bin`` (type: ``String``, *Do not set manually*)

  The state and target bin of the hand.
  Mainly used for parallel activation of dual arms.
