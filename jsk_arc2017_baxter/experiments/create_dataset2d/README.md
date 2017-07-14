# create_dataset2d


## Collect raw data

```bash
roslaunch jsk_arc2017_baxter baxter.launch moveit:=false
roslaunch jsk_arc2017_baxter create_dataset2d_rawdata_main.launch
rosrun jsk_arc2017_common view_dataset2d.py $dirname
```


## Annotate

```bash
dirname=raw_data_$(date +%Y%m%d_%H%M%S)
mv ~/.ros/jsk_arc2017_baxter/create_dataset2d/left_hand/* $dirname
mv ~/.ros/jsk_arc2017_baxter/create_dataset2d/right_hand/* $dirname
rosrun jsk_arc2017_common annotate_dataset2d.py $dirname
rosrun jsk_arc2017_common view_dataset2d.py $dirname
```
