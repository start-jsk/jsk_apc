# Create 2d dataset for object segmentation


## Collect raw data on shelf bins

```bash
roslaunch jsk_arc2017_baxter baxter.launch moveit:=false
roslaunch jsk_arc2017_baxter create_dataset2d_rawdata_main.launch
rosrun jsk_arc2017_common view_dataset2d.py ~/.ros/jsk_arc2017_baxter/create_dataset2d/right_hand
```

![](https://user-images.githubusercontent.com/4310419/28227820-ac70352c-6916-11e7-8a95-277f913cd9e9.gif)


## Collect raw data on tote bin

```bash
roslaunch jsk_arc2017_baxter baxter.launch moveit:=false pick:=false
roslaunch jsk_arc2017_baxter stereo_astra_hand.launch
roslaunch jsk_arc2017_baxter create_dataset2d_rawdata_main.launch box:=tote
rosrun jsk_arc2017_common view_dataset2d.py ~/.ros/jsk_arc2017_baxter/create_dataset2d/right_hand
```


## Annotate

```bash
dirname=raw_data_$(date +%Y%m%d_%H%M%S)
mv ~/.ros/jsk_arc2017_baxter/create_dataset2d/left_hand/* $dirname
mv ~/.ros/jsk_arc2017_baxter/create_dataset2d/right_hand/* $dirname
rosrun jsk_arc2017_common annotate_dataset2d.py $dirname
rosrun jsk_arc2017_common view_dataset2d.py $dirname
```

![](https://user-images.githubusercontent.com/4310419/28228470-a51d903c-6919-11e7-97b1-688f7f1ccf48.gif)
