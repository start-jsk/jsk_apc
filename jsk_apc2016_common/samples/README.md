samples
=======

**samples** includes files used for picking an APC object from table by Fetch.


Usage
-----

```
$ roslaunch jsk_apc2016_common sample_fetch_start.launch
```

Then type this command in another terminal.

```
$ $(rospack find jsk_apc2016_common)/samples/sample-fetch-pick.l
```

If you want to change the target item using GUI, type this in another terminal and change the value of /label_to_mask/label_value.

```
$ rosrun rqt_reconfigure rqt_reconfigure
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
$ roslaunch jsk_apc2016_common play.launch bagfile:=`pwd`/.ros/fetch_demo.bag  ### defined in record.launch
$ rviz -d $(rospack find jsk_apc2016_common)/samples/config/sample_object_segmentation_3d.rviz
```
