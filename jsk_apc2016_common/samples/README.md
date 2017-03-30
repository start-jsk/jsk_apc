samples
=======

**samples** includes files used for Amazon Robotics Challenge. 


Usage
------

```
$ roslaunch sample_fetch_start.launch
$ ./sample-fetch-pick.l
```


Record
-------

```
$ roslaunch record.launch
```


Play
-------

```
$ rossetlocal
$ roslaunch play.launch bagfile:=`pwd`/(file name).bag
$ rviz -d default.rviz
```
