samples
=======

**samples** includes files used for picking an APC object from table by Fetch.


Usage
-----

```
$ roslaunch jsk_apc2016_common sample_fetch_start.launch
$ $(rospack find jsk_apc2016_common)/samples/sample-fetch-pick.l
```


Record
------

```
$ roslaunch jsk_apc2016_common record.launch
```


Play
----

```
$ rossetlocal
$ roslaunch jsk_apc2016_common play.launch bagfile:=`pwd`/fetch_demo.bag  ### defined in record.launch
$ rviz -d $(rospack find jsk_apc2016_common)/samples/config/sample_object_segmentation_3d.rviz
```
