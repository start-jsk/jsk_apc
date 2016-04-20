Stable Achievement of 30 points with new gripper
================================================

- Opend: 2016-04-20
- Deadline: 2016-05-07


Goal
----

Achieve 30 points with new gripper stably.


Configuration
-------------

- Gripper: vacuum2016 (**feature**)
- Item: apc2015
- Hand Camera: old


System
------

Recognition
+++++++++++

1. Location of shelf: old
2. Object recognition in bin: old
3. Grasp plannning in bin: old
4. Detection of grasps with vacuum sensor: new (**feature**)
5. In-hand object recognition: old
6. In-hand detection of grasps: old

Motion
++++++

1. Pick: old
2. Return: (**feature**)
  - into the back of the shelf.
  - when failed to solve ik, replay executed trajectory reversely.

